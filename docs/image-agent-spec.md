# Image-Generation Agent — Architecture Teardown & Build Spec

How ChatGPT's image feature is actually put together, and a concrete spec for
building your own. Written to drop into Iris, but the spec is backend-agnostic.

---

## Part 1 — What you're actually reverse-engineering

The single most useful realization: **"ChatGPT image generation" is two
separate systems**, and people conflate them.

| Layer | What it is | Can you reproduce it? |
|---|---|---|
| **The model** (`gpt-image-1`) | Closed-weights natively-multimodal transformer | **No.** Not published, not reproducible. Substitute it. |
| **The agent around it** | Routing, prompt compilation, reference handling, session state, delivery | **Yes — and this is ~90% of what makes the UX feel good.** |

Almost everything people admire about ChatGPT's image feature — that it
understands "no, make *her* jacket red", that it keeps a character consistent
across turns, that it renders legible text, that edits chain naturally — comes
from the second layer plus the model's *instruction following*. Swap in a
different backend and keep the agent layer, and you keep most of the magic.

### 1.1 The model layer (what's known vs. inferred)

Be careful here — OpenAI has published capabilities, not architecture. Confidence labels:

| Claim | Confidence |
|---|---|
| It's **natively multimodal**, not an LLM calling a separate diffusion model | **Confirmed.** This is the headline change from DALL·E 3, where GPT wrote a prompt and handed it to a detached model. Now image and text live in one model. |
| Image output is **token-based**, generated in the same autoregressive stream as text | **Confirmed** in substance by OpenAI's own framing. |
| Internals are **Transfusion-style** — one transformer, shared attention/FFN, causal masking across text, bidirectional attention *within* image blocks, diffusion-style decoding to pixels | **Inferred.** Consistent with published research and observed behavior, but not officially confirmed. Don't build assumptions on it. |
| API surface: text+image in → image out, up to 16 reference images, up to 4096×4096, multi-turn via `previous_response_id` | **Confirmed** (documented API). |

**Why "native" matters, practically.** Because image tokens flow through the
same weights as text, the model inherits the LLM's world knowledge and
instruction following. That's the direct cause of the two things older
diffusion pipelines were worst at:

- **Legible text inside images** — it's predicting glyphs with a language
  model's priors, not hallucinating letter-shaped texture.
- **Compositional prompt adherence** — "three items, the middle one blue" works
  because the same machinery that parses that sentence is generating the image.

### 1.2 The rendering question specifically

You asked about rendering. Two distinct things:

**Token → pixel decode.** The model emits latent image tokens; a decoder turns
them into RGB. This is where resolution and fidelity are set, and it's a
separate stage from generation — worth keeping separate in your own design too.

**The progressive reveal** (the blurry top-to-bottom sharpening in the UI) is
a *streaming* artifact, not a rendering secret. Autoregressive generation in
raster order means early tokens = top of image. Decode partials as they arrive
and you get exactly that reveal. OpenAI's API exposes this as partial images.
In your build:

- API backends: request partial/streamed images.
- Diffusion backends: emit intermediate denoising steps (step 8, 16, 24 of 30).
- Either way: **it's a UX affordance you implement in the delivery layer**, not
  a property of the model. Copy it — perceived latency drops enormously.

---

## Part 2 — The agent spec

This is the part you build. Seven components.

```
user turn ──▶ ① Router ──▶ ② Prompt Compiler ──▶ ④ Backend Adapter ──▶ ⑥ Delivery
                  │              ▲                      │                   │
                  │              │                      ▼                   ▼
                  └──▶ ③ Reference Resolver      ⑦ Safety Gate       ⑤ Session Graph
                                 ▲                                          │
                                 └──────────────────────────────────────────┘
```

### ① Turn Router

Classifies what the turn wants. This is a cheap LLM call (or the main model
with a forced-choice tool), not a keyword matcher — "can you make that pop
more?" has no keywords.

```
IN:  user_message, session_graph_summary
OUT: { intent, target_node_id | null }
     intent ∈ generate | edit | variation | inpaint | upscale | answer
```

**Design notes.** `answer` matters: "what font is that?" about a generated
image is a vision question, not a generation request. Routing everything to
the generator is the most common failure in homemade versions of this.
`target_node_id` resolves *which* image an edit refers to — usually the most
recent, but "go back to the second one" must work.

### ② Prompt Compiler

Turns conversational intent into a generation prompt. The single highest-leverage
component you own.

```
IN:  user_message, intent, conversation_history, style_memory, target_node
OUT: { prompt, negative_prompt?, params: {size, quality, n} }
```

**Responsibilities:**
- **Resolve deixis.** "make it warmer" → the full prompt of the target node
  with a color-temperature modifier, not the literal string "make it warmer".
- **Carry forward.** An edit prompt must restate everything that should *stay*
  the same, or backends will drift. This is why naive multi-turn editing
  degrades: each turn silently drops constraints.
- **Apply persistent style.** Pull the user's standing preferences from agent
  memory (Iris already has this — `memory_read`).
- **Do not over-embellish.** DALL·E-era prompt rewriting inflated prompts
  heavily because the model was weak at instruction following. Modern backends
  punish this — it overrides user intent. Expand references, not adjectives.

### ③ Reference Resolver

Decides which images condition this turn.

```
IN:  intent, target_node, user_attachments, character_registry
OUT: { reference_images: [...], mask? , conditioning_mode }
```

`conditioning_mode` ∈ `none | style | subject | structural | inpaint`. These
are genuinely different operations and backends expose them differently
(IP-Adapter, ControlNet, reference-image arrays, character embeddings).

**Character registry** is worth building even for a hobby version: a named
store of identity references, so "draw Ziya at a café" pulls the locked
reference automatically. You're already doing this manually with the Higgsfield
Soul system — this formalizes it.

### ④ Backend Adapter (pluggable)

One interface, several implementations. Keep this boundary clean — it's what
lets you swap models as the field moves, which in image generation is every
few months.

```python
class ImageBackend(Protocol):
    name: str
    capabilities: set[str]   # {"edit","inpaint","reference","stream","4k"}

    def generate(self, prompt, *, size, n, references=None,
                 mask=None, seed=None, on_partial=None) -> list[ImageResult]: ...
```

`on_partial` is the streaming hook that gives you the progressive reveal.
`capabilities` lets the router degrade gracefully instead of erroring when a
backend can't inpaint.

### ⑤ Session Graph  ← *the part most builds get wrong*

**Image history is a DAG, not a list.** Every generated image is a node with a
parent, the prompt that made it, the seed, and the backend. Branching is the
normal case: users generate four, pick one, edit it twice, go back and edit the
original differently.

```
node = { id, parent_id, prompt, params, seed, backend,
         file_path, thumbnail, created_at, user_label? }
```

If you store a flat list, "go back to the version before you added the hat"
becomes unanswerable — and that request is extremely common. A DAG makes it a
parent-pointer walk.

### ⑥ Delivery

- **Stream partials** (see 1.2) — biggest perceived-latency win available.
- **Persist** to disk with the full node record as sidecar metadata. Being able
  to reproduce an image from its record is what makes the system feel reliable.
- **Provenance:** attach C2PA content credentials. Most hosted APIs embed these
  already; strip-on-purpose is a bad default and increasingly a legal exposure.
- **Return a node id** with every image so the next turn can reference it.

### ⑦ Safety Gate

Configure this to your platform's policy and your backend's terms. Two things
regardless of policy: the hosted backends enforce their own rules server-side
(you get refusals, not bypasses — design for graceful handling), and if you run
open weights locally you own the entire liability, so an explicit written policy
plus input/output checks is the minimum, not optional polish.

---

## Part 3 — Backend options

| Tier | Option | Notes |
|---|---|---|
| **Hosted, best quality** | `gpt-image-1` (OpenAI) | The thing you're cloning. Native multimodal, 16 refs, 4096², multi-turn state. Pay per image. |
| **Hosted, already wired for you** | **Higgsfield** (MCP, connected) | You already have this. Soul/character system solves identity consistency, plus video. **Start here** — zero integration cost. |
| **Hosted, cheap** | Gemini image models | Strong editing, competitive pricing. |
| **Open weights — text rendering** | **Qwen-Image** (20B MMDiT) | Best open model for legible in-image text, the classic diffusion weakness. |
| **Open weights — unified** | **BAGEL** (ByteDance, 7B active/14B) | Closest open analog to the native-multimodal architecture: understanding *and* generation in one model. |
| **Open weights — unified AR** | **HunyuanImage-3.0**, Janus-Pro-7B, Show-o2 | Native autoregressive text+image in one framework. Research-grade but the right architectural family. |
| **Open weights — workhorse** | FLUX, SDXL | Mature ecosystem, ControlNet/IP-Adapter/LoRA. Best if you want deep control knobs. |

**Hardware reality check for local:** BAGEL-class unified models want ~24GB+
VRAM at usable speed; Qwen-Image 20B more. Check what your machine has before
committing to the local path — a hybrid (hosted generation, local orchestration)
is usually the pragmatic answer.

---

## Part 4 — Build order

Each milestone is independently useful; don't build 1–7 before testing.

1. **M1 — Straight-through.** Backend adapter + one hosted backend + a
   `generate_image` tool in Iris. Prompt goes straight through. *You have image
   generation in your CLI and on Telegram.*
2. **M2 — Session graph + edit.** Node storage, `edit_image` against the last
   node. *Multi-turn editing works.*
3. **M3 — Router + prompt compiler.** Now "make it warmer" works without the
   user restating everything. *This is the jump that makes it feel like ChatGPT.*
4. **M4 — Reference resolver + character registry.** Named identity locks.
   *Consistent characters across sessions.*
5. **M5 — Streaming delivery.** Partial images. *Feels fast.*
6. **M6 — Second backend.** Prove the adapter boundary by swapping. *Now you're
   model-independent, which is the whole point of owning the agent layer.*

---

## Part 5 — Tool contracts (Iris format)

Drop-in shapes matching `iris/tools.py`:

```python
{
  "name": "generate_image",
  "description": (
    "Generate an image from a text description. Returns a node id that later "
    "turns can reference. Use when the user asks for a picture, illustration, "
    "logo, mockup, or any visual artifact."
  ),
  "input_schema": {
    "type": "object",
    "properties": {
      "prompt":     {"type": "string", "description": "Full standalone description. Resolve all pronouns and references — the backend sees no conversation history."},
      "size":       {"type": "string", "enum": ["1024x1024", "1536x1024", "1024x1536"]},
      "references": {"type": "array", "items": {"type": "string"},
                     "description": "Node ids or file paths to condition on."},
      "n":          {"type": "integer", "description": "1-4, default 1."}
    },
    "required": ["prompt"]
  }
}

{
  "name": "edit_image",
  "description": (
    "Modify an existing generated image. Restate the FULL desired result, not "
    "just the change — anything omitted may drift."
  ),
  "input_schema": {
    "type": "object",
    "properties": {
      "node_id": {"type": "string", "description": "Image to edit. Omit for most recent."},
      "prompt":  {"type": "string", "description": "Full description of the desired result."},
      "mask":    {"type": "string", "description": "Optional path to a mask for inpainting."}
    },
    "required": ["prompt"]
  }
}
```

Note how much design intent lives in the *descriptions* — "resolve all
pronouns", "restate the full result". With a strong model driving the tools,
well-written descriptions do a large share of the prompt-compiler's job for
free. That's the cheapest version of component ②, and a fine place to start.

---

## Sources

- [Aman's AI Journal — GPT-4o Native Image Generation](https://aman.ai/primers/ai/gpt4o-native-image-generation/)
- [MarkTechPost — Transformer Meets Diffusion: the Transfusion architecture](https://www.marktechpost.com/2025/04/06/transformer-meets-diffusion-how-the-transfusion-architecture-empowers-gpt-4os-creativity/)
- [OpenAI — Image generation guide](https://developers.openai.com/api/docs/guides/image-generation) · [gpt-image-1 model](https://developers.openai.com/api/docs/models/gpt-image-1) · [Create image edit](https://developers.openai.com/api/reference/resources/images/methods/edit)
- [BAGEL — ByteDance unified multimodal model](https://github.com/bytedance-seed/BAGEL)
- [BentoML — Open-source image generation models, 2026](https://www.bentoml.com/blog/a-guide-to-open-source-image-generation-models)
- [LearnOpenCV — GPT-4o image generation guide](https://learnopencv.com/gpt-4o-image-generation/)

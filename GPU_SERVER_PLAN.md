# GPU Server Plan for Call Audits — Briefing for Decision

Prepared 2026-10-21. This file is a briefing for a reviewer and their AI assistant. It covers what we run today, what we plan to run, the hardware needed, and the rent vs buy options. The reviewer decides **rent or buy, and how many GPUs**. Figures marked *(estimate)* are not measured yet. Prices marked *(AceCloud site, 2026-10-21)* were read from acecloud.ai that day.

---

## 1. The goal

- Audit **40,000 calls per day**: Kataria's ~10,000 plus three other departments of similar size.
- **Day D's calls are audited on day D+1**, so the window is about 20–24 hours.
- Calls must run **in parallel**, at least 4 at a time; the more the better.
- The other departments will use **different AI models** from Kataria's.
- AceCloud (Indian GPU cloud, Noida) offers a **free demo for 1–2 months**. After that we either rent or buy.
- Budget is not the main constraint. The aim is the right-sized choice.
- The current DGX stays, for development and testing.

## 2. What the system does (per call)

1. **Fetch** the call recording and its CRM data. The source is the company CRM, reached over a VPN.
2. **Transcribe**:
   - speaker separation (pyannote);
   - speech-to-text (NVIDIA NeMo Conformer, with a language-model decoder that runs on CPU);
   - Agent/Customer role labelling (rules first, with a small LLM, Qwen3-8B, as fallback).
3. **Audit** with an LLM:
   - **Qwen3-32B** plus our **fine-tuned LoRA adapter** rates 16 soft-skill parameters;
   - stock Qwen3-32B handles the remaining parameters and the booking checks;
   - ratings are read from answer probabilities ("Brio" scoring), not free text, which is faster;
   - result: a score out of 100, an Excel register, and an evidence file.
4. **Dashboard** (Flask) to browse, run batches and export.

**Accuracy today:** 81.5% agreement with human auditors, up from 78.3% before the fine-tune.

## 3. Today's hardware and measured speed

**Hardware:** 1× NVIDIA DGX Spark: GB10 chip, 128 GB unified memory, 20 CPU cores, ARM64.

**Models loaded** (Ollama):

| Model | Size |
|---|---|
| qwen3:32b | 26 GB |
| audit-lora-v1 (the fine-tune, a separate full copy) | 26 GB |
| qwen3:8b (role labelling) | 6.3 GB |

**Measured over about 6,000 real calls:**

| Step | Median | Mean | Slowest 10% |
|---|---|---|---|
| Audit total | 11.6 s | 15.3 s | 25.9 s |
| – stock LLM part | 4.9 s | 8.6 s | 18.3 s |
| – fine-tuned LLM part | 4.8 s | 5.1 s | 6.7 s |
| Transcription | 0.5 s | 0.7 s | 1.4 s |

The transcription figure looks too low to be real; re-measure it in the demo.

**By call type** (audit only):

| Type | Average |
|---|---|
| Follow-up | 10.6 s |
| Service Not Required | 10.7 s |
| Escalation | 16.9 s |
| Booking | 29.9 s |
| **Overall** | **16.9 s** |

**Bottleneck:**
- About 90% of audit time is the LLM.
- Ollama runs **one call at a time**.
- The DGX's memory bandwidth (273 GB/s) limits LLM speed.
- **Maximum today: about 5,000–5,500 calls/day**, which isn't enough even for Kataria's 10k.

## 4. What we will run on the new server

### Software changes (needed whatever hardware is chosen)

1. **Replace Ollama with vLLM** for the audit model:
   - many calls processed at once on each GPU;
   - the model stored in 8-bit FP8, with the fine-tuned LoRA loaded as an adapter, so one copy of the base model is shared;
   - other departments' fine-tunes on the same base can be added as further adapters at little memory cost;
   - vLLM returns the answer probabilities Brio needs (top-20), and reuses the shared system prompt between calls (prefix caching).
2. **Parallel worker pool** in the pipeline engine, which is one call at a time today:
   - N transcription workers;
   - M audit workers.
3. **Accuracy check:** re-run our 545-call test set on the FP8 model and confirm agreement stays at about .815, as with the current setup.
4. **Throughput benchmark:** 1,000 real calls; measure calls/hour per GPU and the real transcription time.

### Calls in parallel, per H100 80 GB *(estimate)*

**Memory budget:**

| Item | Size |
|---|---|
| Usable GPU memory (90%) | ~72 GB |
| Qwen3-32B FP8 + adapter | ~33 GB |
| Overhead | ~4 GB |
| **Left for calls in progress** | **~35 GB** |

The model's KV cache, the working memory each call in progress needs, takes about 256 KB per token in normal precision (16-bit). That's about 3.1 GB for a full 12k-token call, or about 1.8 GB for an average ~7k-token call.

| Working memory (KV cache) kept at | Full-length calls (12k) | Average calls (~7k) |
|---|---|---|
| Normal precision (16-bit) | ~11 at once | ~19 at once |
| Compressed (FP8) | ~22 at once | ~38 at once |

**Plan:** start at **16 audits at once per GPU**, then tune toward 30–40.

### Layout on a 4× H100 node

**Option 1: all GPUs on audit** (best for measuring)

| GPU | Runs | At once |
|---|---|---|
| Each of the 4 | One vLLM copy of the audit model, plus ~2 transcription workers in the remaining memory | ~16 audits (30–40 tuned) + 2 transcriptions |
| **Node total** | | **~64 audits (120–160 tuned) + ~8 transcriptions** |

**Option 2: split by department**

| GPU | Runs |
|---|---|
| 0–1 | Kataria audit, two vLLM copies with load shared between them (~32 at once, 60–80 tuned) |
| 2 | The other departments' models |
| 3 | Transcription, 4–8 workers |

Running in parallel doesn't make each call faster. A call may take 40–60 s instead of 15 s, but many finish together.

## 5. Capacity *(estimate, to be confirmed by the demo)*

**Per call** *(estimate)*: about 15–25k tokens of input across the audit's LLM calls, and a few hundred tokens of output.

**One H100 with vLLM:** about **20,000–30,000 calls/day** for this pipeline. That's input-heavy work bound by compute, at about 50–60% real-world efficiency.

| GPUs (H100) | Calls/day *(estimate)* | Covers 40k? |
|---|---|---|
| 1 | 20k–30k | No |
| 2 | 40k–60k | Just; little headroom |
| 3 | 60k–90k | Yes, with headroom |
| 4 | 80k–120k | Yes, 2–3× headroom; room for other models |

**Other GPUs compared with an H100 for this workload** *(estimate)*:

| GPU | Compared with H100 | Notes |
|---|---|---|
| RTX PRO 6000 Blackwell (96 GB) | ~0.5× | Lower compute; lots of memory |
| H200 (141 GB) | ~1.0–1.2× | Same compute, more memory and bandwidth |
| L40S (48 GB) | ~0.4× | A 32B model is a tight fit |
| A100 (80 GB) | ~0.5× | No FP8 |

## 6. Option A — Rent (AceCloud, Noida)

*(AceCloud site, 2026-10-21; INR per month; 6-month terms 5% off, 12-month terms 10% off)*

| GPU | 1× | 2× | 4× | Hourly (1×) |
|---|---|---|---|---|
| H100 HGX 80 GB | ₹1,80,000 | ₹3,60,000 | ₹7,20,000 (104 vCPU, 1 TB RAM) | not listed |
| H200 NVL 141 GB | ₹2,00,498–2,38,309 | ₹4,32,671–4,76,583 | ₹8,65,307 | ₹343–408 |
| RTX PRO 6000 96 GB | ₹1,21,599–1,65,150 | ₹2,86,701–3,30,252 | ₹5,73,354–6,60,456 | ₹208–283 |
| L40S 48 GB | ₹83,000–1,00,876 | ₹1,83,110–2,01,708 | ₹4,03,372 | — |
| A100 80 GB | ₹90,000 | ₹1,80,000 | ₹3,60,000 | — |
| L4 24 GB | ₹36,730–48,193 | ₹84,875–96,338 | ₹1,69,702 | — |

**Recommendation for the free demo:** the **4× H100 HGX node (`N.HGXH100.1000`)**, plus 1× RTX PRO 6000 for a side-by-side comparison. AceCloud is offering **H200 + RTX PRO 6000** instead; that works well too (see below).

**Ask AceCloud for:**
- Ubuntu 22.04 or 24.04, with an NVIDIA driver for CUDA 12.4 or newer;
- Docker with the NVIDIA container toolkit;
- root access, and a fixed public IP;
- permission to run an OpenVPN client (a TUN device), needed to reach the CRM;
- about 2 TB of NVMe storage (recordings are ~50 GB a day at 40k calls);
- hourly H100 pricing after the demo;
- data-protection terms in writing.

### Demo hardware AceCloud is offering: H200 + RTX PRO 6000

AceCloud's GPU page lists H200 141 GB from ₹2,20,000/month and RTX PRO 6000 96 GB from ₹95,636/month *(acecloud.ai/gpu/nvidia-cloud, 2026-10-21)*.

| | H200 141 GB | RTX PRO 6000 96 GB |
|---|---|---|
| Speed for this workload compared with H100 *(estimate)* | ~1.0–1.2× | ~0.5× |
| Room for calls in progress (Qwen3-32B in FP8) | ~100 GB, so ~30 full-length / ~50 average calls at once | ~60 GB, so ~19 full-length / ~33 average calls at once |
| Can run the model at full 16-bit precision (BF16, ~64 GB, as trained) | Yes, ~70 GB left for calls | Yes, but only ~20 GB left (few calls) |
| Calls/day per GPU *(estimate)* | ~20k–35k | ~10k–15k |
| Best demo role | Kataria audit model (vLLM) | Transcription, the role-labelling model, and a second audit copy or another department's model |

**Demo test:** run the same 1,000 calls on each GPU separately to get calls/day per GPU for both, then both together for the full pipeline. One H200 plus one RTX PRO 6000 is about 1.5× an H100, around 30k–45k calls/day for Kataria *(estimate)*. That's close to 40k, with no room for the other departments' models.

## 7. Option B — Buy our own server *(prices are rough estimates for India; get vendor quotes)*

**Spec matching the rented 4× H100 node:**

| Part | Spec |
|---|---|
| GPUs | 4× H100 80 GB SXM with NVLink, e.g. Dell PowerEdge XE8640 or a Supermicro 4-GPU HGX system |
| CPUs | 2× 32-core Xeon or EPYC |
| RAM | 1 TB DDR5 |
| Storage | 2× 3.84 TB NVMe, plus 8–16 TB for recordings |
| Network | 2× 10/25 GbE |
| Power | Redundant supplies, ~4–5 kW at full load |
| Warranty | 3-year on-site |

**Purchase options with similar capacity:**

| Option | Rough price *(estimate)* | Notes |
|---|---|---|
| 4× H100 SXM (HGX) | ₹1.3–1.8 crore | Same GPUs as the rental node |
| 4× H100 NVL / H200 NVL (PCIe) | ₹1.1–1.5 crore | More memory per GPU, similar speed here |
| 8× RTX PRO 6000 Blackwell | ₹0.8–1.0 crore | Cheapest; 768 GB of GPU memory; speed must be checked in the demo |

**Running costs** *(estimate)*: about ₹40,000–80,000 a month for power and cooling (≈3.5 kW continuous at ~₹9 per unit, plus AC), plus staff time for maintenance. The server also needs a dedicated power line, a UPS and a cooled room, or rented data-centre rack space.

## 8. Rent vs buy side by side (4× H100 level)

| | Rent (AceCloud) | Buy |
|---|---|---|
| Upfront | ₹0 | ₹1.3–1.8 crore *(estimate)*; lower if the GST is claimed back |
| Monthly | ₹7.2 lakh (₹6.48 lakh on a 12-month term) | ~₹0.4–0.8 lakh, power and cooling |
| Pays for itself | — | ~21–32 months *(estimate)* |
| 5-year total | ~₹3.9 crore (12-month terms) | ~₹1.6–2.3 crore *(estimate)*: purchase plus running costs |
| Customer data | Leaves the company; **needs company approval** | Stays in-house |
| CRM / VPN access | VPN from the cloud server | Same as today on the local network |
| Scaling | Add or remove GPUs month to month | Fixed; buy more to grow |
| Ready in | Days | Weeks to months (H100 lead times) |
| Upkeep | Provider | Our team: failures, drivers, power, cooling |
| Ageing | Upgrade freely | New GPU generations in 2–3 years make it look slow |

**At the 2× H100 level:** rent ₹3.6 lakh a month vs buy ~₹0.7–0.9 crore *(estimate)*. The payback period is about the same.

## 9. Demo plan and decision rule

**During the demo:**
1. Set up the pipeline on the cloud node with vLLM and the parallel workers.
2. Run the 545-call accuracy check on the FP8 model. The target is agreement ≥ .81.
3. Run 1,000+ real calls with 16, 24, 32 and 40 audits at once per GPU. At each step, record calls/hour and GPU memory use, and stop raising it when calls/hour stops rising.
4. Measure the real transcription time per call.
5. Run the same test on 1× RTX PRO 6000.

**Then decide:**
1. GPUs for Kataria = 40,000 ÷ (measured calls per day per GPU), plus 30–50% headroom.
2. Add the GPUs the other departments' models need. Each different base model needs its own memory, and a large one may need a whole GPU.
3. Round up.
4. Rent or buy that count.

| Demo shows per GPU | Kataria needs | Likely total to get |
|---|---|---|
| ~30,000/day | 2 | 2× H100, if other models fit alongside |
| ~20,000/day | 2–3 | 4× H100 |
| ~12,000/day | 4 | 4× H100 or more |

## 10. Open questions and risks

- **Data approval:** customer recordings and transcripts on a third-party cloud need company approval before real calls go there.
- **Other departments' models:** which models are they, how big, and are they fine-tuned on the same Qwen3-32B base? This decides how many extra GPUs are needed.
- **Transcription time:** the measured 0.5 s looks too low and must be re-measured. If it's really 3–5 s per call, transcription needs its own GPU capacity.
- **FP8 accuracy:** the current model runs in a different 4-bit format (Q4_K_M), so FP8 must be re-validated on the test set.
- **Software effort:** the vLLM move and the parallel workers are real development work that has to be done before the benchmark.
- **CPU side:** the speech-to-text language-model decoding runs on CPU, so the node needs enough cores. The 4× node's 104 vCPUs should be ample.

## 11. Notes for the AI assistant reviewing this

- **Decision to support:** rent vs buy, and how many GPUs of which type, for 40,000 calls/day with a D+1 window.
- **Firm numbers:** the measured DGX timings in section 3 and the AceCloud prices in section 6.
- **Estimates:** throughput per GPU, parallel-call counts, purchase prices and running costs. Challenge them.
- **Not in this file:** the codebase, credentials, network details and customer data are deliberately left out.

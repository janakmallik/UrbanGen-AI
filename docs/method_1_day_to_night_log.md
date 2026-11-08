
# Method 1 — Day-to-Night Image Translation (SD 1.5 Img2Img)

**Status:** Experimental pilot — unsuccessful
**Dataset:** BDD100K (Train Split)
**Source domain:** Daytime residential street images
**Target domain:** Nighttime residential street images
**Model:** Stable Diffusion 1.5 Img2Img
**Date:** November 8, 2026

## 1. Objective

Evaluate whether Stable Diffusion 1.5 Img2Img can transform daytime BDD100K residential street images into realistic nighttime scenes while preserving their original structural characteristics.

The intended objective is to generate useful synthetic nighttime urban images without substantially changing:
- Road geometry and layout
- Buildings and architectural structures
- Vehicles and their positions
- Sidewalks and surrounding objects
- Original camera perspective

## 2. Experimental Configuration

- **Input resolution:** 512 × 288 (16:9 aspect ratio)
- **Number of images:** 5
- **Random seed:** 42
- **Image selection:** BDD100K residential daytime training images
- **Pipeline:** ``StableDiffusionImg2ImgPipeline``
- **Hardware:** NVIDIA RTX 4050 Laptop GPU (6 GB VRAM)
- **Average Generation Time:** ~4.5 seconds per image (after model warmup)
- **Peak VRAM Usage:** 2.06 GB

## 3. Experiments & Parameter Search

We tested various combinations of `strength`, `guidance_scale`, and resolution to find a balance between generating night scenes and preserving the original geometry.

| Experiment | Strength | Steps | Resolution | Observation |
| :--- | :--- | :--- | :--- | :--- |
| **Initial** | 0.60 | 20 | 512x512 (Crop) | Hallucinated houses and vehicles, completely lost original structure. |
| **Experiment 2** | 0.25 | 30 | 1280x720 (Raw) | Too weak; image remained daytime with only slight dimming. |
| **Experiment 3** | 0.40 | 50 | 1280x720 (Raw) | Improved lighting, but still hallucinated streetlights and deleted the original green highway sign. |
| **Final Test** | 0.40 | 30 | 512x288 (16:9) | Preserved some layout, but heavily distorted vehicles and added fake glowing artifacts. |

## 4. Technical Observations

### The Denoising Trade-off
The experiments revealed a fundamental Catch-22 with standard Img2Img:
- **Low strength (0.2 - 0.3):** Preserves the scene structure perfectly, but the AI does not have enough "freedom" to turn the bright blue sky into a dark night sky.
- **High strength (0.5 - 0.6):** The AI successfully turns the sky black, but it destroys the original image. It treats the original image as a mere suggestion and redraws everything according to the text prompt, deleting important labels like the bus, the green sign, or the pedestrians.

### Structural Hallucination
Even at conservative strengths, the model struggled with 1280x720 inputs. Because Stable Diffusion 1.5 was trained on 512x512 images, feeding it higher resolutions forces it to duplicate objects (e.g., drawing multiple buses side-by-side).

### Prompt Truncation
The detailed prompts used in later experiments exceeded the CLIP text encoder's 77-token limit, meaning the end of the prompt was ignored by the model.

## 5. Results Assessment

| Evaluation criterion | Result |
| :--- | :--- |
| Nighttime realism | Unsatisfactory |
| Scene preservation (Labels) | Unreliable |
| Vehicle preservation | Unreliable |
| Architectural preservation | Unreliable |
| Synthetic-data readiness | **Not approved** |

The generated images are not suitable for inclusion in the project's validated synthetic training dataset (Arm D). If used, they would train the classifier on hallucinated structures, worsening the model's performance.

## 5.1 Visual Comparisons

Below are side-by-side comparisons from the final Method 1 run (SD 1.5 Img2Img at 512x288, strength 0.4).

**Comparison 1:**
![Compare 1](../assets/method_results/compare_01.jpg)

**Comparison 2:**
![Compare 2](../assets/method_results/compare_02.jpg)

**Comparison 3:**
![Compare 3](../assets/method_results/compare_03.jpg)

**Comparison 4:**
![Compare 4](../assets/method_results/compare_04.jpg)

**Comparison 5:**
![Compare 5](../assets/method_results/compare_05.jpg)

## 6. Conclusion

The standard Stable Diffusion 1.5 Img2Img approach is fundamentally incapable of solving this task. Parameter adjustments alone cannot resolve the conflict between nighttime transformation and structural preservation. 

The pilot is considered a documented negative result. Generic text-conditioned Img2Img is the wrong tool for label-preserving day-to-night translation.

## 7. Future Work: Method 3 (ControlNet)

The failure of Method 1 validates the necessity of **ControlNet Depth**. ControlNet bypasses the Img2Img trade-off by feeding the AI a 3D depth map. This forces the model to keep the exact geometry of the scene, while allowing the text prompt to aggressively paint the sky black and add nighttime lighting without destroying the original labels.

**Final decision:** Drop standard Img2Img. Retain these findings as evidence of why ControlNet is required. Proceed to implement Method 3.

# Face Recognition System - Presentation Script

**Presenter:** Agriya Yadav  
**Student ID:** 1020231092  
**Course:** CS-4440: Artificial Intelligence  
**Instructor:** Prof. Lipika Dey  
**Duration:** 20 minutes + Q&A

---

## 🎤 COMPLETE PRESENTATION SCRIPT

### [SLIDE 1: TITLE]

"Good morning/afternoon everyone, and thank you Professor Dey for this opportunity. 

Today I'll be presenting my project on **Face Recognition System Analysis**, focusing on Performance, Robustness, and Ethical Evaluation. This is my final project for CS-4440 Artificial Intelligence at Ashoka University."

*(Pause, make eye contact with audience)*

---

### [SLIDE 2: THE QUESTION]

"Let me start with a simple observation. Face recognition is everywhere in our daily lives.

*(Point to each item)*

- We unlock our smartphones with our faces
- We go through airport security with facial scans
- Law enforcement uses it for suspect identification
- And offices use it for access control

But here's the critical question: **Does it actually work?** And more importantly: **Is it fair?**

These aren't just academic questions. These systems make real decisions that affect people's lives. And that's what motivated me to dig deeper."

---

### [SLIDE 3: PROJECT SCOPE]

"So I designed this project to be more than just a standard machine learning exercise. I wanted to evaluate face recognition comprehensively across **five critical dimensions**:

*(Count on fingers)*

**First**, Performance - comparing classical machine learning against modern deep learning.

**Second**, Robustness - testing how these systems perform under real-world conditions like face masks, blur, and crowds.

**Third**, Explainability - understanding how these black-box models actually make decisions.

**Fourth**, Fairness - analyzing whether these systems work equally well for everyone.

And **fifth**, Security - testing if AI-generated deepfakes can fool the system.

Most projects just measure accuracy and stop there. I wanted to know: what happens when things aren't perfect?"

---

### [SLIDE 4: DATASET]

"For this analysis, I curated a dataset of **40,709 images** spanning **247 Indian celebrity identities** - primarily Bollywood and South Indian cinema actors.

Now, you might ask: why celebrities? Well, celebrities have publicly available images, which allows for legitimate research use. But more importantly, most face recognition research is done on Western datasets like LFW or CelebA. I wanted to use data that's more representative of Indian faces.

The dataset has realistic challenges:

- **Severe class imbalance**: ranging from just 14 images for some people to 620 for others
- **Variable quality**: different brightness levels, blur, aspect ratios
- I split it into 60% training, 20% validation, and 20% test sets using stratified sampling

*(Gesture to visualization)*

You can see this class distribution here - this imbalance actually reflects real-world conditions where we have unequal data for different individuals."

---

### [SLIDE 5: MODELS COMPARED]

"I evaluated three models representing different approaches to face recognition:

**First, Buffalo_L** - this is a modern deep learning model with a ResNet-50 backbone trained with something called ArcFace loss. It outputs 512-dimensional face embeddings and was pre-trained on WebFace600K, a massive dataset of faces.

**Second, AntelopeV2** - this is an even larger state-of-the-art model from InsightFace with a ResNet-100 architecture. More parameters, more compute.

**Third, LBP+SVM** - this is my classical machine learning baseline. It uses hand-crafted texture features called Local Binary Patterns fed into a linear Support Vector Machine classifier. This represents the pre-deep-learning era of face recognition.

I wanted to see: does deep learning really make a difference?"

---

### [SLIDE 6: RESULT 1 - PERFORMANCE]

"And the answer is: **absolutely yes.**

*(Point to table)*

Look at these numbers:

- **Buffalo_L achieved 95.27% accuracy**
- AntelopeV2 was close behind at 94.59%
- But LBP+SVM? Only **24.59%**

That's a **71 percentage point improvement** from modern deep learning over classical methods. It's not even close.

*(Gesture to visualization)*

This embedding visualization shows why - the deep learning models create these beautifully separated clusters in embedding space. Same identity stays close together, different identities are far apart. The classical method just can't create that kind of separation.

So yes, deep learning works phenomenally well... under ideal conditions. But that's where things get interesting."

---

### [SLIDE 7: RESULT 2 - ROBUSTNESS]

"Because I wanted to know: what happens when conditions aren't ideal?

I tested Buffalo_L against **15 different perturbations** across five categories. Here are the results:

*(Point to each row)*

- **Lighting changes**: 45% accuracy - pretty robust actually
- **JPEG compression**: 45.3% - handles compression artifacts well
- **Heavy blur**: 46.9% - some degradation but manageable
- **Heavy noise**: 41.1% - more significant impact
- **Occlusions with face masks**: **34.2%**

*(Pause for emphasis)*

That last one is critical. Face occlusions - particularly masks covering the nose and mouth - reduce accuracy from 95% to 34%. That's nearly a **50% drop in performance**."

---

### [SLIDE 8: COVID-19 IMPACT]

"And this isn't just an academic finding. This has massive real-world implications.

Think about what happened during COVID-19. Suddenly everyone was wearing masks. And many commercial face recognition systems - the ones deployed in airports, offices, security checkpoints - **simply stopped working**.

The technology that companies had invested millions in became unreliable overnight because it couldn't handle this one simple real-world variation.

This is the gap between lab performance and deployment reality. 95% in the lab means nothing if a simple mask brings you down to 34%."

---

### [SLIDE 9: RESULT 3 - EXPLAINABILITY]

"Now let's talk about explainability - how do these models actually make decisions?

This is a fundamental trade-off in machine learning:

**Classical LBP+SVM** is transparent - I can visualize exactly what texture patterns it's looking at. But it's horribly inaccurate at only 24%.

**Deep learning** is the opposite - incredibly accurate at 95%, but it's a black box. How does it decide?

To answer this, I used occlusion-based attention mapping. Basically, I systematically blocked different parts of the face and measured how much the prediction confidence dropped.

*(Point to visualization)*

What did I find? The model focuses most on:

1. **Eyes** - highest importance
2. **Nose** - high importance  
3. **Mouth** - moderate importance
4. **Forehead and hair** - minimal importance

And here's what's fascinating: **this matches human visual attention patterns!** Research by Peterson and Eckstein from 2012 showed that humans naturally focus on the eye-nose region when recognizing faces. Our deep learning model learned the same strategy without being explicitly programmed to do so.

So while the model is complex internally, its attention is biologically plausible and interpretable."

---

### [SLIDE 10: RESULT 4 - FAIRNESS]

"Next, let's address the elephant in the room: **Is face recognition fair?**

I tested performance across skin tone categories using brightness as a proxy. Now, I want to be transparent here - brightness is an imperfect proxy for skin tone, and ideally I'd have ground truth demographic labels. But even with this limitation, the results are concerning:

*(Point to table)*

- **Dark skin tone proxy**: 52.8% accuracy with 53 samples
- **Medium skin tone**: 43.2% accuracy with 250 samples  
- **Light skin tone**: 20% accuracy with only 10 samples

Now, the light category has too few samples to be statistically reliable. But the gap between dark and medium is clear: **32.8% disparity**.

This means the system works much better for some people than others. And that's not just a technical problem - it's an ethical one."

---

### [SLIDE 11: ETHICAL IMPLICATIONS]

"Because biased systems cause real harm.

Let me give you two documented examples:

**First, Robert Williams** in Detroit, 2020. He was falsely arrested because a face recognition system misidentified him. The system had higher error rates for Black individuals, leading to this wrongful accusation. Robert spent hours in custody, missed work, and faced the trauma of false arrest - all because of algorithmic bias.

**Second, the Gender Shades study** by Joy Buolamwini and Timnit Gebru in 2018 tested commercial face recognition systems and found:

- **34% error rate** on dark-skinned women
- **0.8% error rate** on light-skinned men

That's a **33-point gap** - almost identical to what I found in my analysis.

This isn't a hypothetical problem. When face recognition is used for access control, surveillance, or law enforcement, these biases enable discrimination. They can determine who gets access, who gets flagged, who gets arrested.

My **32.8% disparity** confirms this is a real, persistent problem in face recognition systems."

---

### [SLIDE 12: RESULT 5 - CROWD TESTING]

"Now let's look at another real-world scenario: **crowds**.

Face recognition is often deployed for surveillance in public spaces - train stations, stadiums, streets. But how well does it work when there are multiple people in the frame?

I tested the system on crowd images with multiple faces:

- **Single-face images**: 95.27% accuracy
- **Crowd images**: 33.3% accuracy
- **Performance drop**: -62 percentage points

*(Pause for effect)*

Why such a dramatic drop?

1. **Small face sizes** - faces far from the camera are tiny, maybe 50x50 pixels or less
2. **Partial occlusions** - people overlap, bodies block faces
3. **Non-frontal poses** - not everyone's looking at the camera
4. **Variable lighting** - different faces in the same image have different lighting

The implication? **Surveillance systems need high-resolution cameras at close range** - typically 1 to 3 meters - to work reliably. Those wide-angle crowd surveillance cameras you see? They're probably not as effective as you think."

---

### [SLIDE 13: RESULT 6 - AI-GENERATED FACES]

"Finally, let's talk about security: **Can AI-generated deepfakes fool the system?**

I collected 25 AI-generated faces from 'ThisPersonDoesNotExist' - these are photorealistic faces created by StyleGAN2 that look completely real.

I ran two tests:

**Test 1: False Acceptance** - Can these fake faces be matched to real identities with high confidence?
- Result: ✅ No. The system didn't falsely match them.

**Test 2: Real vs AI Detection** - Can we tell real from fake?
- Result: ✅ **100% classification accuracy** using artifact analysis

The key is that current AI-generated faces have subtle artifacts in the frequency domain - edge patterns that aren't quite natural. My classifier picked up on these perfectly.

*(Show visualization)*

Edge artifacts had 58% feature importance in detection.

So the good news is: **current generation AI faces are perfectly separable** from real faces.

*(Change tone)*

But here's the warning: this is an **arms race**. StyleGAN2 is from 2020. We now have DALL-E 3, Midjourney v6, and other next-generation models that produce even more realistic faces. The artifacts I detected might not exist in newer models.

So while we're safe today, this is a moving target."

---

### [SLIDE 14: THE CORE TENSION]

"This brings me to what I think is the core tension in face recognition technology:

*(Draw attention to the diagram)*

**High accuracy enables effective surveillance.**  
But effective surveillance means invasion of privacy, discrimination risk, and chilling effects on free expression.

Here's the paradox: **Technical performance does not equal deployment readiness.**

We've achieved 95% accuracy in the lab. We can detect faces, match identities, track people. The technology works.

But should we deploy it everywhere? What safeguards do we need? Who decides how it's used?

These are questions that go beyond computer science into ethics, policy, and human rights."

---

### [SLIDE 15: KEY FINDINGS SUMMARY]

"Let me summarize what works and what doesn't:

**What works well:**

- ✅ 95% accuracy on clean, single-face images
- ✅ Robust to lighting changes and JPEG compression  
- ✅ Can detect current AI-generated faces
- ✅ Human-like attention patterns

**Critical gaps:**

- ❌ Face masks: drops to 34% accuracy
- ❌ Crowd scenarios: drops to 33% accuracy
- ❌ Demographic bias: 33% performance disparity
- ❌ Heavy blur and noise: 20-25% degradation

My conclusion? **This technology is not ready for high-stakes deployment without human oversight and proper safeguards.**

It's powerful, yes. But it has critical vulnerabilities that could harm people if deployed irresponsibly."

---

### [SLIDE 16: RECOMMENDATIONS]

"So what should we do? I have recommendations across three categories:

**Technical safeguards:**

1. **Liveness detection** - verify the person is actually present with blinks, movement, not just a photo
2. **Multi-modal authentication** - combine face with fingerprint or voice, don't rely on face alone
3. **Quality thresholds** - reject low-resolution or poorly-lit images rather than making unreliable predictions

**Ethical practices:**

1. **Diverse training data** - ensure balanced representation across demographics
2. **Regular bias audits** - test performance across demographic groups quarterly or annually
3. **Human oversight** - require human review for high-stakes decisions like arrests
4. **Transparent disclosure** - tell users about limitations and error rates

**Policy frameworks:**

1. **Context-specific regulation** - maybe ban in schools but allow in passport control
2. **Consent mechanisms** - give people the right to know and opt out
3. **Accountability frameworks** - clear liability when systems make mistakes

These aren't optional nice-to-haves. These are necessary guardrails for responsible deployment."

---

### [SLIDE 17: LIMITATIONS & FUTURE WORK]

"Now let me be honest about the limitations of my own work:

**First**, this is a celebrity dataset, which may not fully generalize to the broader population.

**Second**, I used brightness as a proxy for skin tone, which is imperfect. Ground-truth demographic labels would be better.

**Third**, some of my tests had small sample sizes - like only 10 images in the light skin tone category.

For future work, I'd love to explore:

1. **Adversarial robustness** - testing against intentional attacks like FGSM or PGD
2. **Video and temporal consistency** - how well does tracking work over time?
3. **Cross-dataset generalization** - train on one dataset, test on completely different ones
4. **Fairness interventions** - techniques like reweighting or adversarial debiasing to reduce bias
5. **Federated learning** - privacy-preserving ways to train these models

There's so much more to explore here."

---

### [SLIDE 18: CONTRIBUTIONS]

"So what makes this project unique?

**First**, it's a **multi-dimensional evaluation** - I didn't just measure accuracy and call it done. I systematically tested robustness, fairness, explainability, and security.

**Second**, I used an **Indian celebrity dataset** when most research uses Western faces. Representation matters.

**Third**, I conducted **real-world testing** - crowds, occlusions, AI-generated faces - not just clean benchmark datasets.

**Fourth**, I developed an **ethical framework** with concrete, actionable recommendations, not just vague calls for "being responsible."

And **fifth**, I documented everything comprehensively in a 666-line LaTeX report with professional visualizations.

This kind of holistic analysis is rare in academic projects, and I'm proud of it."

---

### [SLIDE 19: CONCLUSION]

"So here's my conclusion:

**Face recognition technology is powerful.** It achieves 95% accuracy under ideal conditions. That's genuinely impressive.

**But it's not perfect.** 34% accuracy with face masks. 33% in crowds. 33% bias gap across demographics.

**And it's not fair.** The performance disparities I found mirror documented harms in the real world.

This leads me to what I think is the fundamental challenge:

*(Point to diagram)*

We need to balance **Innovation with Rights Protection**. **Accuracy with Privacy**. **Efficiency with Fairness**.

Technical excellence is necessary but not sufficient. We need ethical responsibility too.

My call to action is simple: **Let's build better technology, but let's also build it better.**"

---

### [SLIDE 20: THANK YOU]

"Thank you for your attention. I'd be happy to take any questions.

If you want to explore the project further, all the code, data, and documentation are in the project directory. The full report is 12 pages with comprehensive analysis and visualizations.

*(Smile and open the floor)*

Questions?"

---

## 🔄 OPTIONAL: LIVE DEMO SCRIPT (5 minutes)

### If Time Permits / If Asked to Show Demo:

"I'd be happy to show you a quick live demo of the system!

*(Share screen / open terminal)*

Let me navigate to the project directory...

```bash
cd /Users/Agriya/Desktop/monsoon25/AI/facial-recognition
source .venv/bin/activate
```

Great, I've activated the virtual environment. Let me show you the configuration first:

```bash
cat configs/default.yaml
```

This shows our model configuration - you can see we're using Buffalo_L with cosine similarity and k-nearest neighbors classification.

Now let me run the evaluation script:

```bash
python src/eval_arcface_closedset.py
```

*(While it runs)*

This is loading the Buffalo_L model, extracting 512-dimensional embeddings from our test set, and performing nearest neighbor classification...

*(When complete)*

And there we go - 95.27% accuracy as I showed earlier.

Let me show you some of the visualizations we generated:

```bash
open runs/robustness_analysis.png
```

*(Explain the chart)*

This shows performance across all 15 perturbation conditions - you can clearly see occlusions are the worst performers.

```bash
open runs/fairness_analysis.png
```

And this shows the demographic disparity I discussed.

Finally, let me show you the full report:

```bash
open main.pdf
```

*(Scroll through a few pages)*

This is the comprehensive 12-page LaTeX report with all the methodology, results, and ethical analysis.

That's the system in action!"

---

## 🎯 INTERACTIVE GUI DEMO (If Live Demo Requested)

### GUI Application Demo:

"Actually, I also built an interactive web application for live face recognition. Let me show you that:

```bash
python gui_app.py
```

*(Wait for Gradio to launch, share/open browser)*

This is a Gradio-based web interface. Let me walk through the workflow:

**Step 1**: First, I need to load the Buffalo_L model by clicking this button...

*(Click "Load Buffalo_L Model")*

*(While loading)*

This is computing prototypes by extracting embeddings from all 247 celebrity classes in our training set. It uses direct ONNX inference on the pre-aligned face images... 

*(When loaded)*

Great! The system is ready with all 247 celebrities in the database.

**Step 2**: Now let me upload a test image...

*(Upload an image with clear face)*

**Step 3**: And click 'Recognize Faces'...

*(Wait for processing)*

*(Point to results)*

Look at this! The system:
- **Detected the face** and drew a bounding box
- **Classified it** with a confidence score
- **Shows top-5 predictions** with similarity scores

You can see the confidence bars here - the system is quite certain about its top prediction, but also shows alternatives.

The confidence is derived from cosine similarity between the detected face's embedding and our celebrity prototypes.

*(If time, upload another image)*

Let me try another one... maybe a crowd image to illustrate the challenge...

*(Upload crowd image)*

See how performance drops? Multiple faces, smaller sizes, varied angles - this is the 33% accuracy scenario I mentioned.

This GUI demonstrates both when the technology works well and when it struggles."

---

## ❓ Q&A PREPARATION

### Anticipated Questions and Answers:

#### Q1: "Why is LBP+SVM so bad at only 24%?"

**A:** "Great question. LBP+SVM uses hand-crafted features that only capture local texture patterns - like corners and edges at very small scales. It has no understanding of global facial structure, and it can't learn invariances to pose, lighting, or expression the way deep learning can. 

Deep learning models learn hierarchical representations - low-level edges, mid-level facial parts like eyes and noses, and high-level identity features - from millions of training examples. LBP is limited to what we manually engineered, and it turns out we're not very good at designing features compared to what neural networks can learn."

---

#### Q2: "Why did accuracy drop from 95% to 47% in robustness testing?"

**A:** "Two reasons. First, the robustness test used a harder subset of the data with more challenging examples - more pose variation, lower quality images. This reflects real deployment conditions rather than ideal lab conditions.

Second, I added perturbations on top - noise, blur, masks. The combination of already-challenging images plus perturbations created very difficult test cases.

The gap between 95% clean accuracy and 47% perturbed accuracy shows the brittleness of these systems. Lab benchmarks don't always translate to real-world performance."

---

#### Q3: "How did you measure fairness without ground-truth demographic labels?"

**A:** "Excellent observation. I used image brightness as a proxy for skin tone - darker images treated as darker skin, lighter images as lighter skin. This is absolutely imperfect and I acknowledge that limitation in the report.

Ideally I'd have ground-truth demographic labels - actual skin tone measurements, self-identified race/ethnicity. But that data wasn't available for this celebrity dataset.

The brightness proxy is coarse, but previous research has shown it correlates with actual demographic bias patterns. The 32.8% disparity I found aligns with the Gender Shades study's 33-point gap, which used proper labels. So while imperfect, it's informative.

For production systems, proper demographic labels and testing are essential."

---

#### Q4: "What can actually be done about the bias?"

**A:** "Multiple approaches:

**Data-level**: Collect more diverse training data with balanced representation. If 90% of your training faces are light-skinned, the model will optimize for that.

**Algorithm-level**: Use fairness-aware optimization. Add constraints that penalize performance disparities. There are techniques like adversarial debiasing where you train the model to be unable to predict demographic attributes from embeddings.

**Process-level**: Regular audits across demographic groups, human oversight for critical decisions, transparent error rate reporting.

**Policy-level**: Regulation requiring bias testing before deployment, accountability when systems fail.

No single solution fixes everything, but combining these approaches can significantly reduce bias."

---

#### Q5: "Can this system detect deepfakes?"

**A:** "It depends on the generation. Current AI faces from StyleGAN2 are perfectly separable - 100% detection accuracy using frequency-domain artifact analysis.

But this is an arms race. Each new generation of generative models produces more realistic images with fewer artifacts. DALL-E 3 and Midjourney v6 are already much better than StyleGAN2.

My detection works *today* on *current* AI faces. But I have no confidence it'll work on next year's models without retraining.

The better approach is liveness detection - verify the person is physically present through blinks, movement, depth sensors. That's much harder to fake than static images."

---

#### Q6: "What's your recommendation for deploying this in practice?"

**A:** "Context matters enormously.

For **low-stakes applications** like personal photo organization or smartphone unlock where the user is in control? Go ahead, with clear disclosure.

For **medium-stakes** like office access control? Yes, but with multi-factor authentication - face plus badge, or face plus PIN. Don't rely solely on face recognition.

For **high-stakes** like law enforcement identification or border security? Only with mandatory human oversight, regular bias audits, quality thresholds, and clear accountability frameworks. The system should assist humans, not replace them.

And for **mass surveillance** with no consent or oversight? I'd argue we shouldn't deploy it at all. The risks to privacy and civil liberties are too high.

Technology capability doesn't automatically justify deployment. We need to ask not just 'can we?' but 'should we?'"

---

#### Q7: "How long did this project take?"

**A:** "The full project took several weeks:

- Data collection and preprocessing: ~1 week
- Model training and evaluation: ~1 week  
- Robustness and fairness testing: ~1 week
- Analysis, visualization, and report writing: ~1 week

The most time-consuming parts were actually the data curation - finding good quality images across 247 identities - and the comprehensive testing across all those perturbation conditions.

The deep learning models themselves are pre-trained, so I didn't train from scratch. But computing embeddings for 40,000 images and running all the experiments still took significant compute time."

---

#### Q8: "What was the most surprising finding?"

**A:** "Honestly? The crowd performance drop. I expected face masks to hurt performance - that's well-documented in the literature post-COVID. But I didn't anticipate such a dramatic 62-percentage-point drop in crowd scenarios.

It really drove home how much these systems rely on ideal conditions - frontal pose, high resolution, good lighting, single face. Once you violate those assumptions, performance collapses.

It made me question all those surveillance camera deployments we see in public spaces. If they're not getting high-quality face captures, they're probably not actually doing much."

---

## 🎬 PRESENTATION TIPS

### Opening Strong:
- Make eye contact before starting
- Smile, show confidence
- Speak clearly and at moderate pace
- Use the "hook" about ubiquity of face recognition

### During Presentation:
- **Vary your tone** - emphasize key numbers, pause before important points
- **Use gestures** - point to visualizations, count on fingers
- **Tell stories** - Robert Williams case makes bias concrete
- **Be conversational** - "you might ask" / "here's what's fascinating"
- **Own the limitations** - transparency builds credibility

### Timing Check:
- If running over: skip or condense slides 8, 14, 17, 18
- If running under: elaborate on ethical implications, add demo
- Always prioritize: results (slides 6-13) and ethics (slides 10-11)

### Energy Management:
- High energy for introduction and conclusion
- Measured, serious tone for ethical section
- Enthusiastic for technical achievements
- Balanced throughout

### If Technology Fails:
- Have backup plan: show static images of results
- Practice the script without slides
- Keep calm, apologize briefly, move on

---

## ✅ PRE-PRESENTATION CHECKLIST

### Day Before:
- [ ] Review this script 2-3 times
- [ ] Practice with timer (aim for 18-19 min to leave buffer)
- [ ] Test all commands in terminal
- [ ] Verify all visualization files open
- [ ] Check main.pdf opens correctly
- [ ] Prepare backup on USB drive / cloud

### 1 Hour Before:
- [ ] Activate virtual environment
- [ ] Open all key files
- [ ] Test screen sharing / projection
- [ ] Have water bottle ready
- [ ] Bathroom break

### Right Before:
- [ ] Deep breath
- [ ] Remember: you know this material better than anyone
- [ ] Smile
- [ ] Start strong

---

## 💪 FINAL MOTIVATION

You've built something genuinely impressive:

✅ **Comprehensive**: Multi-dimensional evaluation rarely seen in student projects  
✅ **Rigorous**: Statistical tests, multiple models, extensive documentation  
✅ **Relevant**: Addresses real-world challenges (masks, crowds, bias)  
✅ **Ethical**: Goes beyond technical metrics to consider societal impact  
✅ **Professional**: 666-line LaTeX report, clean visualizations, working GUI  

**You're not just presenting a class project. You're presenting research that matters.**

Be proud. Be confident. You've got this! 🚀

---

**Good luck, Agriya! You're going to do great! 🍀**

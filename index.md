---
layout: page
title: Hello
subtitle: We Are Team Speak Up
sitemap:
  priority: 0.9
---

<img src="{{ '/assets/img/pudhina.jpg' | prepend: site.baseurl }}" id="about-img">

<div id="describe-text">
	<p>Our Mission is to Democratize Audio Production with Machine Learning</p>
	<p>check out our project <strong> <a href="https://github.com/speakupai"> repository</a> </strong></p>
</div>

# The Team

|Matt Linder&nbsp;&nbsp;&nbsp;|[![Linkedin](https://i.stack.imgur.com/gVE0j.png)](https://www.linkedin.com/in/matt-linder-ml/)|&nbsp;&nbsp;|[![GitHub](https://i.stack.imgur.com/tskMh.png)](https://github.com/mholmeslinder)|

|Rana Ahmad&nbsp;&nbsp;&nbsp;|[![Linkedin](https://i.stack.imgur.com/gVE0j.png)](https://www.linkedin.com/in/ranataimurahmad/)|&nbsp;&nbsp;|[![GitHub](https://i.stack.imgur.com/tskMh.png)](https://github.com/taimur1871)|

|Wilson Ye&nbsp;&nbsp;&nbsp;|[![Linkedin](https://i.stack.imgur.com/gVE0j.png)](https://www.linkedin.com/in/wilsonye1/)|&nbsp;&nbsp;|[![GitHub](https://i.stack.imgur.com/tskMh.png)](https://github.com/LqYe)|

# Background and Significance of Project

Audio is invisible. On its face, this might seem like the most obvious statement in the world - humans literally can't see sound (unless they have synesthesia...). But what I mean is this: humans are visual-dominant creatures and because we can't see audio, it’s easy to miss its importance - and its impact. This impact is felt in so many areas of media, but let's scope it just to spoken audio. [Research has shown that people are more likely to trust information that comes to them as higher quality spoken audio.](https://news.usc.edu/141042/why-we-believe-something-audio-sound-quality/) 

Whether they're aware of it or not, people are more likely to consume content that *sounds better*, but the problem is - it's difficult and time-consuming to make spoken audio sound professional. 

Instagram, Snapchat, Tiktok, and YouTube have revolutionized photo and video content creation - giving consumers access to filters, effects, and editing techniques that used to be the realm of professionals. The same revolution has not yet come for spoken audio. 

There are plenty of DAW plugins and other products that can produce amazing results, but given the fact that maybe only 5-10% of my audience understood that 'DAW' stands for 'Digital Audio Workstation', it's clear that these products are out of the realm of your typical consumer. The vast majority of content that has professional-sounding audio is either recorded by professionals or by amateurs who have spent large amounts of time and money learning to record and engineer audio.

This is where SpeakUpAI comes in. We want every aspiring podcaster, YouTuber, MOOCs teacher, and every other creator of spoken audio content to be able to produce professional-sounding audio. With SpeakUpAI's state of the art machine learning technology, users can simply record spoken audio on their phone or tablet, upload it to our site, and our ML model will return an 'enhanced' version - denoised, dereverberated, normalized, EQ'd, etc, etc. - for them to download and use in whatever application they choose.

SpeakUpAI wants everyone to sound professional. We're *hear* to democratize audio.

---



# Explanation of Data sets

We've intentionally sourced the same three datasets that were used in the creation of [HiFi-GAN](https://daps.cs.princeton.edu/projects/HiFi-GAN/index.php?env-pairs=DAPS&speaker=f10&src-env=all), the academic paper upon which we've based our research. 

## Speech - DAPS

As the creators of the [DAPS dataset](https://archive.org/details/daps_dataset) describe it:

> A dataset of professional production quality speech and corresponding aligned speech recorded on common consumer devices.
[...]
This dataset is a collection of aligned versions of professionally produced studio speech recordings and recordings of the same speech on common consumer devices (tablet and smartphone) in real world environments. It consists of 20 speakers (10 female and 10 male) reading 5 excerpts each from public domain books[...]

(They even have [a paper](https://ccrma.stanford.edu/~gautham/Site/Publications_files/mysore-spl2015.pdf) they wrote about it!)

This is a dataset that is more or less tailor-made for our application - supervised learning for speech denoising. We have high-quality studio recordings of many different speakers (10 male and 10 female) for training.  The unaltered audio (in .wav format) of these recordings acts as our output/target (or y), and we perform pre-processing and convolution on that same audio with our **Noise** and **Impulse Response** datasets to create our input (or X).

Additionally, we have the same twenty speakers reading the same material on different consumer devices in different real-world environments for testing. This really is perfect for our application, since it allows us to test generalization on never-before-seen examples while still maintaining conceptual continuity for subjective testing. These test examples also very closely mirror our intended use-case for our model - processing audio recorded across different devices in many different environments.

If our model generalizes well to this test set, but performs significantly worse on other real-world examples, it will give us a much clearer idea of whether we're overfitting on the 18 specific speakers in our training set. In turn, this will allow us to better tweak our data collection and pre-processing to make the model more robust.

## Noise - ACE Challenge dataset

Our "Noise" dataset is a collection of audio recordings focusing on what would typically be called background or ambient noise - the hisses, hums, and other persistent sounds present in almost all non-professional audio recordings. Because our audio targets are (intentionally) devoid of such noise, we use this dataset to add it back in and create our input data.

In this case, we've sourced the [ACE Challenge dataset](https://ieee-dataport.org/documents/ace-challenge-2015#files).

## Impulse Response (Room sound/reverb) - REVERB Challenge dataset

Another important aspect of audio recordings - and one that's particularly noticeable in professional recordings in particular - is the impulse response or 'room sound' of the space in which the recording is made. This is often referred to in the audio industry as 'reverb' (short for reverberation), and most professional voice-over, podcast, and other spoken media go through great lengths to minimize the audience's perception of the room in which the media was recorded (this can be referred to as making a recording 'dry').

Because our output audio from the DAPS set is already very dry, we create our inputs by convolving the output with a variety of impulse responses taken from the [REVERB Challenge dataset](https://www.openslr.org/28/).

---

# Explanation of Processes (Methods)

## HiFi-GAN

As the authors [explain](https://daps.cs.princeton.edu/projects/HiFi-GAN/index.php?env-pairs=DAPS&speaker=f10&src-env=all): 

> Real-world audio recordings are often degraded by factors such as noise, reverberation, and equalization distortion. This paper introduces HiFi-GAN, a deep learning method to transform recorded speech to sound as though it had been recorded in a studio. We use an end-to-end feed-forward WaveNet architecture, trained with multi-scale adversarial discriminators in both the time domain and the time-frequency domain. It relies on the deep feature matching losses of the discriminators to improve the perceptual quality of enhanced speech. The proposed model generalizes well to new speakers, new speech content, and new environments. It significantly outperforms state-of-the-art baseline methods in both objective and subjective experiments.

Here, generator G includes a feed-forward WaveNet for speech enhancement, followed by a convolutional Postnet for cleanup. Discriminators evaluate the resulting waveform (Dw, at multiple resolutions) and mel-spectrogram (Ds).

On a practical level, we're using a PyTorch implementation of the HiFi-GAN model, [forked](https://github.com/w-transposed-x/hifi-gan-denoising) from the awesome folks at `w-transposed-x`. As with any deep learning projects, our initial efforts went towards getting the model up and running (aka lots of troubleshooting), test runs, and hyperparameter tweaks based on performance. 

We did initial testing on Google Colab, which is great for testing since it's an inexpensive platform. The disadvantage, however, is that users are limited to VMs with a single GPU. 

So, once we got hyperparameters and performance to a functional level, we moved to Amazon EC2 (+ S3 for storage). EC2 instances have the advantage of being scalable - since our PyTorch implementation was built for distributed training on up to 4 GPUs, we spun up a 4GPU instance and started testing.

Currently, we're dealing with performance scaling issues between our single GPU training on Colab and the four GPU EC2.

## Pipeline

Currently, work on HiFi-GAN is limited to audio recorded at 16khz sample rate. True high-fidelity audio is recorded at sample rates of 44.1khz or above. To this end, we've also implemented an [Audio Super Resolution](https://openreview.net/pdf?id=S1gNakBFx) U-Net model, with the intention of training it on 44.1k audio downsampled to 16k.

Once we have this model trained, we can use it after HiFi-GAN in a pipeline to take raw speech audio, enhance/denoise it, and then upsample it to the desired high-fidelity sample rate.

Currently, we have our super resolution U-Net properly implemented in TensorFlow 2.x (it was originally implemented in 1.x, and the porting process was time-consuming), though further training work will wait as we optimize HiFi-GAN.

---

# Explanation of Outcomes (current results)

We take a small clip of audio from our DAPS dataset, recorded on a common consumer device (iPad) in a common location (office), and we run it through our HiFi-GAN archicture.

The following results are from HiFi-GAN  after a very, very brief training (~1.5% of total steps from the original paper).

### Before HiFi-GAN

### After

[{{site.url}}/project-update-media/Week_12_raw_snippet.wav]({{site.url}}/project-update-media/Week_12_raw_snippet.wav)

## Subjective Results

[{{site.url}}/project-update-media/Week_12_denoised_snippet.wav]({{site.url}}/project-update-media/Week_12_denoised_snippet.wav)

As you can hear, almost 100% of the background noise (often described as 'hum' or - in this case - 'hiss') has been removed. There is still some obvious artifacting in the spoken voice, but remember - this is a preliminary with a VERY tiny fraction of the full training.

## Objective Results

![{{site.url}}/project-update-media/Untitled.png]({{site.url}}/project-update-media/Untitled.png)

 

As you can see, the model is improving consistently with each increase in training. We'll be continuing to ramp up the training steps/epochs as we continue development - e.g. iron out some of the kinks with augmentation and distributed training - until our loss curves flatten, at which point we'll consider the results 'final' for this particular model iteration.

---

# System Design and Ethical Considerations

## Overall System Design

![{{site.url}}/project-update-media/Untitled%201.png]({{site.url}}/project-update-media/Untitled%201.png)

You can read in MUCH greater detail about the system design of HiFi-GAN in the [original paper](https://arxiv.org/abs/2006.05694), but in a nutshell:

The model is a GAN architecture using a Wavenet architecture (plus an optional simple convolution-based "Postnet") as the generator network with four(!) discriminator networks. Three of these are waveform-based - one utilizing with audio at original sample rate, and the other two downsampled by factors of 2. The last discriminator is Mel Spectrogram-based, and the adversarial loss is added to a relatively novel concept called "Deep Feature Loss", about which you can read in the original paper.

We chose this architecture because:

1. The results shown on the DAPS dataset were the most impressive we found in our survey of ML-based speech-enhancement.
2. Using heterogenous sources - waveform and Mel Spectrogram - in the adversarial architecture of a GAN seems extremely promising. Most previous audio-based GANS are 100% spectrogram based, and as such, tend to have serious phase-related issues.

## HiFi-GAN System Design

![{{site.url}}/project-update-media/Untitled%202.png]({{site.url}}/project-update-media/Untitled%202.png)
This is subject to change as we develop and iterate our process and deployment, but for the moment, system design looks like:

- A user navigates to the SpeakUpAI website, where they land at the index page. There, they are prompted to upload an audio file for enhancement.
- Once they choose a file and click `Upload`, that file is sent to SpeakUpAI's storage (an S3 bucket, for the moment), and then passed to our HiFi-GAN model for inference.
- When inference is complete, it spits out an audio file (`.wav`), which is then moved to our storage, and the user is prompted to download the enhanced file.
- Once the user downloads their enhanced file, the original and enhanced versions are erased from our storage.

## Ethical Considerations

Since we're designing a fully opt-in system that doesn't request any sensitive information, our ethical considerations fall into two main categories:

1. Data Privacy
2. Bias towards/against certain types of speaking voices/languages

**Data Privacy** is obviously important, since users' audio: a.) is their own creative work and potentially subject to copyright or other intellectual property laws, and b.) could potentially contain sensitive information. 

To that end, it's our mission to make sure that our network and storage systems are as secure as possible, and that users' content is completely erased from our system as soon as they've downloaded their enhanced audio.

**Bias** is always a concern in ML systems, and it takes many forms. But, in this case - we're concerned with bias towards or against certain voices, accents, or languages. At the moment, our scope only gives us time to focus on English speakers, so extending the training to other languages will come later, but part of our current model evaluation will relate to how it is able to generalize to speakers with voices and accents different from those in the training and validation sets.
---

# Future work and Timeplan

## Timeplan

![{{site.url}}/project-update-media/Untitled%203.png]({{site.url}}/project-update-media/Untitled%203.png)

## Future Work

- Implementation of Bryan/HiFi-GAN Impulse Response augmentation
- Longer training of Generator (wavenet and wavenet-postnet) models
- Using Audio Super-Resolution model to upscale results from 16k to 44.1k (and beyond!)
- Taking different audio/video formats as input - currently only `.wav`
- Optimizing inference performance (currently works at near-realtime on a single GPU)
- Implement Spark

# Related Work (Papers, github)

This is by no means comprehensive, but the following are some of the papers, repos, and products we researched while creating SpeakUpAI. Items marked with "***" are directly utilized in our research.

## Automatic Mixing

- [Deep learning and intelligent audio mixing](https://www.eecs.qmul.ac.uk/~josh/documents/2017/WIMP2017_Martinez-RamirezReiss.pdf)
- [Intelligent Sound Engineering](https://intelligentsoundengineering.wordpress.com/category/machine-learning/)

## Vocal DeNoising specific

### Papers

- [Audio Super-Resolution using Neural Nets](https://openreview.net/pdf?id=S1gNakBFx) ***
- [Practical Deep Learning Audio Denoising](https://sthalles.github.io/practical-deep-learning-audio-denoising/)
- [Audio Super-Resolution using neural networks](https://paperswithcode.com/paper/audio-super-resolution-using-neural-networks)
- [Speech Denoising by Accumulating Per-Frequency Modeling Fluctuations](https://paperswithcode.com/paper/audio-denoising-with-deep-network-priors)
- [using Deep learning to reconstruct high-resolution audio](https://blog.insightdatascience.com/using-deep-learning-to-reconstruct-high-resolution-audio-29deee8b7ccd)
    - [Preprocessing tools](https://github.com/jhetherly/EnglishSpeechUpsampler/tree/master/preprocessing)
- [Speech-enhancement (denoising)](https://github.com/vbelz/Speech-enhancement)
- [Conventional DSP Noise Reduction](https://timsainburg.com/noise-reduction-python.html)

### Products

- [Accusonus](https://accusonus.com/products/audio-repair)
- [Izotope Vocal Assistant](https://www.izotope.com/en/learn/how-to-mix-vocals-with-nectar-3-vocal-assistant.html)

## Audio and De-Noising GANs

- [WAVEGAN](https://github.com/chrisdonahue/wavegan) ***
- [HiFi-GAN](https://daps.cs.princeton.edu/projects/HiFi-GAN/index.php?env-pairs=DAPS&speaker=f10&src-env=all) ***
- [Magenta GANSynth](https://magenta.tensorflow.org/gansynth)
- [DN-Gan for Denoising](https://www.sciencedirect.com/science/article/abs/pii/S1746809419302137) and [code](https://github.com/manumathewthomas/ImageDenoisingGAN)

## Impulse Response Augmentation

- [IMPULSE RESPONSE DATA AUGMENTATION AND DEEP NEURAL NETWORKS
FOR BLIND ROOM ACOUSTIC PARAMETER ESTIMATION](https://arxiv.org/pdf/1909.03642.pdf) ***

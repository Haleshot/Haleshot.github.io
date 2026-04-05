---
title: "A weekend at PyConf Hyderabad 2026"
description: "Representing SciPy India at PyConf Hyderabad 2026; community booths, familiar faces, and a few reminders about why in-person OSS spaces still matter."
pubDatetime: 2026-03-16T00:00:00Z
author: "Srihari Thyagarajan"
tags: ["python", "open-source", "community", "conference", "scipy-india", "foss", "meetups"]
featured: false
draft: false
---

This past weekend, I was in Hyderabad for [PyConf Hyderabad](https://2026.pyconfhyd.org/). The main reason for the trip was to represent [SciPy India](https://scipy-india.github.io/); a community I help co-organize along with [Agriya Khetarpal](https://github.com/agriyakhetarpal) and others.

We had signed up as [community partners](https://2026.pyconfhyd.org/#community-partners) for the conference a while back and had bought tickets as well. Later, after getting selected as community partners, we received complimentary passes; we shared those with people we knew in the community who were interested in attending, along with a few discount codes. That felt like a good start even before the event properly began.

This was also my second visit to Hyderabad. The first was back in August last year for the [Open Source Summit India 2025](https://events.linuxfoundation.org/archive/2025/open-source-summit-india/), which I had [posted about earlier](https://www.linkedin.com/posts/srihari-thyagarajan_opensource-linux-summit-activity-7362437980256452609-rc43/). So the city already felt a little familiar this time around; still, it was nice to be back.

On Saturday, Agriya and I also met [Sai Rahul](https://rahulporuri.in/) from FOSSUnited for dinner. We spoke about [SciPy India](https://scipy-india.github.io/), community things, and the usual overlap between conferences and the work that continues after them. No photos from that one, unfortunately; that part will have to stay undocumented.

## At the booth

Most of the conference, for me, revolved around the SciPy India booth and the conversations that happened around it. A lot of people stopped by to ask what SciPy India is, what we have been trying to revive, and where the community is headed next. What I especially liked was that the booth brought together two kinds of conversations: people who remembered the original SciPy India conference and told us they had attended it, and people who were only learning about the community now and wanted to understand how they could help or contribute.

It was also nice seeing familiar faces again. [Jaidev Deshpande](https://2026.pyconfhyd.org/speakers/jaidev-deshpande), for instance, was around; he is someone I have seen across many OSS events in the country, and he had also [run a workshop](https://2026.pyconfhyd.org/speakers/jaidev-deshpande) the previous day. Some of the photos that will eventually show up in the album for this post are thanks to him as well.

<style>
.pyconf-gallery {
  position: relative;
  margin: 1.5rem 0;
}
.pyconf-gallery-scroll {
  display: flex;
  gap: 1rem;
  overflow-x: auto;
  scroll-snap-type: x mandatory;
  scroll-behavior: smooth;
  -webkit-overflow-scrolling: touch;
  padding-bottom: 0.75rem;
  scrollbar-width: thin;
}
.pyconf-gallery-scroll::-webkit-scrollbar {
  height: 6px;
}
.pyconf-gallery-scroll::-webkit-scrollbar-track {
  background: transparent;
}
.pyconf-gallery-scroll::-webkit-scrollbar-thumb {
  background: #888;
  border-radius: 3px;
}
.pyconf-gallery-item {
  flex: 0 0 min(85%, 520px);
  scroll-snap-align: center;
  border-radius: 8px;
  overflow: hidden;
  position: relative;
  background: #111;
}
.pyconf-gallery-item img {
  width: 100%;
  height: 360px;
  object-fit: cover;
  display: block;
  transition: transform 0.3s ease;
}
.pyconf-gallery-item:hover img {
  transform: scale(1.02);
}
.pyconf-gallery-item figcaption {
  padding: 0.5rem 0.75rem;
  font-size: 0.85rem;
  color: #ccc;
  text-align: center;
  background: #111;
}
.pyconf-gallery-hint {
  text-align: center;
  font-size: 0.8rem;
  color: #888;
  margin-top: 0.25rem;
}
</style>

<div class="pyconf-gallery">
  <div class="pyconf-gallery-scroll">
    <figure class="pyconf-gallery-item">
      <img src="/images/pyconf-hyd-2026/group-pic.jpg" alt="Group photo at the PyConf Hyderabad 2026 booth" loading="lazy" />
      <figcaption>At the PyConf Hyderabad booth</figcaption>
    </figure>
    <figure class="pyconf-gallery-item">
      <img src="/images/pyconf-hyd-2026/community-booth-1.jpeg" alt="Conversations at the SciPy India booth" loading="lazy" />
      <figcaption>Conversations at the SciPy India booth</figcaption>
    </figure>
    <figure class="pyconf-gallery-item">
      <img src="/images/pyconf-hyd-2026/scipy-showcase-community-partner.jpeg" alt="Showcasing the SciPy India x BangPypers meetup" loading="lazy" />
      <figcaption>Showcasing the SciPy India x BangPypers meetup</figcaption>
    </figure>
    <figure class="pyconf-gallery-item">
      <img src="/images/pyconf-hyd-2026/parul-pandey-keynote.jpeg" alt="Parul Pandey's keynote on Data Science in the age of LLMs" loading="lazy" />
      <figcaption>Parul Pandey's keynote</figcaption>
    </figure>
    <figure class="pyconf-gallery-item">
      <img src="/images/pyconf-hyd-2026/anand-s-session.jpeg" alt="Anand S's session on how students learn Python" loading="lazy" />
      <figcaption>Anand S on how students learn Python</figcaption>
    </figure>
    <figure class="pyconf-gallery-item">
      <img src="/images/pyconf-hyd-2026/panel-discussion.jpeg" alt="Panel discussion at PyConf Hyderabad 2026" loading="lazy" />
      <figcaption>Panel discussion</figcaption>
    </figure>
  </div>
  <p class="pyconf-gallery-hint">← scroll to see more →</p>
</div>

## Talks I kept thinking about

I particularly liked Parul Pandey's keynote: ["Data Science in the age of LLMs"](https://2026.pyconfhyd.org/speakers/parul-pandey). We had spoken before about notebooks; that is a space I care about a lot (as may be obvious if you have read my [about page](https://haleshot.github.io/about/)), and she had once asked me about [marimo](https://marimo.io/) after seeing some of the community booth work I had done at earlier events like PyCon India and IndiaFOSS. She had also written a thoughtful piece on [switching from traditional notebooks to marimo](https://pandeyparul.medium.com/why-im-making-the-switch-to-marimo-notebooks-6e2218b5c98d). So hearing her keynote landed a little differently for me; there was prior context there, and I appreciated the framing.

I also enjoyed Anand S's [session](https://2026.pyconfhyd.org/speakers/anand-s) on [how students learn Python](https://sanand0.github.io/talks/2026-03-15-how-students-learn-python/), along with the panel discussion earlier. I usually come away learning something from the way he presents, not just from the content itself. More and more, I am realizing that presenting is an art with its own discipline; delivery matters. Since teaching is something I'd like to do more of, especially around notebooks and educational tooling, that session gave me both practical takeaways and a reminder to pay closer attention to how material is taught, not just what is taught.

## Why being there helped

The more I keep attending meetups and conferences (and, apparently, establishing enough of a pattern that people now come up to me and say they'll wait for my LinkedIn post about the event), the more I keep coming back to the same conclusion: these events are really about the people. I remember [Siddharta Govindaraj](https://www.linkedin.com/in/siddharta/) telling me something along these lines at PyCon India as well. At larger events especially, many talks are recorded and can be watched later, even if I still like picking out a few talks, BoFs, or panel discussions ahead of time and sitting through them in person. What does not translate in the same way afterwards is the conversation.

That showed up again here. A lot of SciPy India coordination that would normally take time to get moving on [Zulip](https://scipyindia.zulipchat.com/join/4mesdxfbbpl4titgtdzx4iwv/) became much easier because most of us were in the same place. That is part of why I remain fairly bullish on in-person community spaces; they create momentum.

I have felt this for a while now, and it is one reason we started doing things like the joint [SciPy India x BangPypers meetup](https://scipy-india.github.io/blogs.html?id=scipy-india-x-bangpypers-feb-2026) in February. Hyderabad reinforced that feeling. Being physically present together made it easier to align on a few community-related threads and accelerate some things already in motion. I will leave those specifics for later, but I came back more excited than I was before the trip.

That broader pattern has been visible across other meetups for me as well. Back in January, I gave a talk at [Rust Delhi Meetup #12](https://www.linkedin.com/posts/srihari-thyagarajan_kicking-off-2026-with-rust-delhi-meetup-12-activity-7420450968770764800-SQyb) on building incremental data pipelines with CocoIndex; the slides are [here](https://lnkd.in/gSX3PWQg). Different community, different topic, same conclusion: showing up in person still matters, especially when your work sits somewhere between open source, developer tooling, and community-building.

## Heading back

By the end of it, Agriya and I were on our way back to the airport after what felt like a short but genuinely useful weekend. I came back with a better sense of where some SciPy India conversations stand, a renewed appreciation for the people doing steady community work, and a few small reminders about why I keep making time for these events in the first place.

I've also [cross-posted](https://forum.fossunited.org/t/my-experience-at-pyconf-hyderabad-2026/7549) this on the [FossUnited Discourse forum](https://forum.fossunited.org/). Thanks to FossUnited for supporting this visit; I appreciate the help in making the trip possible.

---
title: "My experience at Open Source Day 2026"
description: "What building around marimo taught me about devtool communities, and why the best community work usually starts with helping users well."
pubDatetime: 2026-04-10T00:00:00Z
author: "Srihari Thyagarajan"
tags: ["open-source", "community", "developer-relations", "developer-advocacy", "conference", "marimo", "devtools"]
featured: false
draft: false
---

I put in a talk proposal for [Open Source Day 2026](https://osd.opensourceweekend.org/) because I had been sitting on a bunch of thoughts about community work that still felt fairly fresh, and the community track seemed like a good place to turn them into something more coherent. Most of those thoughts came out of my time doing dev-rel/advocacy, documentation, and [ambassador work](https://marimo.io/ambassadors) around open-source tools, especially [marimo](https://marimo.io/); I wanted to speak about them while they still felt close to the work itself.

The talk I proposed was ["Lessons from building a devtool community"](https://www.linkedin.com/posts/open-source-weekend_osd26-speakerannouncement-opensource-activity-7440331969390993409-9zbb?utm_source=share&utm_medium=member_desktop&rcm=ACoAADSJzvgBkjBd85IWDyUWA6ttzq8B-NDq-Hs). The organizers had also shared [an introductory speaker post](https://www.linkedin.com/posts/open-source-weekend_osd2026-sriharithyagarajan-opensource-activity-7445432368649289728-CvKV?utm_source=share&utm_medium=member_desktop&rcm=ACoAADSJzvgBkjBd85IWDyUWA6ttzq8B-NDq-Hs) ahead of the event. The event itself was hosted at [Dhirubhai Ambani University](https://www.daiict.ac.in/) in Gandhinagar, Gujarat (my first time in the state).

After checking in and getting my speaker badge, I spent some time walking through campus before the sessions began. The place had peacocks and other bits of wildlife moving through it in a way that made the morning feel calmer than most conference mornings usually do. The image carousel below has a few of those campus photos.

## What I spoke about

My session was about how I got into open-source community work through [marimo](https://marimo.io/), and what that changed in the way I think about contributors, users, and useful work. I did not present it as a playbook to follow. If anything, one of the points I kept returning to was that community work is subjective, context-dependent, and shaped by the product, the people, and the stage a project is in. What I could offer were examples from work I had actually done, and the patterns I thought were worth naming.

One part of that was the code snippet work, which came less from assuming what people needed and more from asking users, watching how they worked, and noticing the patterns they repeatedly used. A lot of people were typing the same boilerplate into cell blocks over and over and would've appreciated a quick way to drop those in without having to think about it. That request [turned into a proper snippets feature](https://github.com/marimo-team/marimo/issues/3602). Examples become much better when they come from actual user contact rather than generic starter material. A user's before-and-after visualization workflow exposed a gap that later became [`mo.image_compare()`](https://github.com/marimo-team/marimo/pull/5091). Over time, that kind of work changed the way I judged contributions.

That was really the argument underneath the talk: durable devtool communities tend to grow from close product contact, teaching artifacts, attribution, and small habits that compound. In the talk, I showed that work in a more quantified way across documentation, tutorials, contributor spotlights, booth conversations (e.g., [PyCon India booth](https://www.linkedin.com/posts/srihari-thyagarajan_just-wrapped-up-a-nice-weekend-at-pycon-india-activity-7373595613570961408-yfoz?utm_source=share&utm_medium=member_desktop&rcm=ACoAADSJzvgBkjBd85IWDyUWA6ttzq8B-NDq-Hs), [IndiaFOSS booth](https://www.linkedin.com/feed/update/urn:li:activity:7376141032054427649/?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base%3Bc3vHJyMNSDyfhgy%2F9Mpf6g%3D%3D)), issue triage, and a lot of public-facing glue work that does not always look glamorous but matters later.

I also spent some time on the teaching and attribution side of this. One of the slides was just the words "Teach in public" -- that section was probably the one I spent the most time on. A lot of the community work I found most meaningful did not look like "growth" in the usual sense; it looked like making useful notebooks, building out learning material, writing up community projects, and properly crediting the people doing interesting work. It also meant working with professors and students around pedagogy itself; helping them think about the computation tools they were already using, how newer tools were evolving, and how that could improve the way they taught. In the talk, I tied that back to people whose work and framing I genuinely look up to, including [Greg Wilson](https://third-bit.com/), along with real advocates rather than hypothetical ones; for instance, [Parul Pandey's write-up on switching to marimo](https://pandeyparul.medium.com/why-im-making-the-switch-to-marimo-notebooks-6e2218b5c98d) and the way she now shares data science work using it publicly. Over time, some of those contributors stayed close to the tool, brought it into their teams, or started advocating for it on their own. Trust, usefulness, attribution, and good examples tend to carry further than forced enthusiasm. The [slides are up here](https://haleshot.github.io/talks/open-source-weekend-04-2026/) if you want to follow the thread.

<style>
.osd-gallery {
  position: relative;
  margin: 1.5rem 0;
}
.osd-gallery-scroll {
  display: flex;
  gap: 1rem;
  overflow-x: auto;
  scroll-snap-type: x mandatory;
  scroll-behavior: smooth;
  -webkit-overflow-scrolling: touch;
  padding-bottom: 0.75rem;
  scrollbar-width: thin;
}
.osd-gallery-scroll::-webkit-scrollbar {
  height: 6px;
}
.osd-gallery-scroll::-webkit-scrollbar-track {
  background: transparent;
}
.osd-gallery-scroll::-webkit-scrollbar-thumb {
  background: #888;
  border-radius: 3px;
}
.osd-gallery-item {
  flex: 0 0 min(85%, 520px);
  scroll-snap-align: center;
  border-radius: 8px;
  overflow: hidden;
  position: relative;
  background: #111;
}
.osd-gallery-item img {
  width: 100%;
  height: 360px;
  object-fit: cover;
  display: block;
}
.osd-gallery-item.contain img {
  object-fit: contain;
  background: #fff;
}
.osd-gallery-item video {
  width: 100%;
  height: 360px;
  object-fit: cover;
  display: block;
}
.osd-gallery-caption {
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  background: linear-gradient(transparent, rgba(0,0,0,0.7));
  color: white;
  padding: 1rem 0.75rem 0.75rem;
  font-size: 0.8rem;
}
</style>

<div class="osd-gallery">
  <div class="osd-gallery-scroll">
    <div class="osd-gallery-item contain">
      <img src="/images/osd-2026/osd-2026-speaker-pass.png" alt="OSD 2026 speaker badge" loading="lazy" />
      <div class="osd-gallery-caption">Speaker badge, Dhirubhai Ambani University</div>
    </div>
    <div class="osd-gallery-item">
      <img src="/images/osd-2026/osd-2026-talk.jpg" alt="Talk session at OSD 2026" loading="lazy" />
      <div class="osd-gallery-caption">Lessons from building a devtool community</div>
    </div>
    <div class="osd-gallery-item">
      <img src="/images/osd-2026/osd-2026-campus-wildlife.jpg" alt="Wildlife on campus at DAIICT" loading="lazy" />
      <div class="osd-gallery-caption">Campus wildlife, DAIICT Gandhinagar</div>
    </div>
    <div class="osd-gallery-item">
      <img src="/images/osd-2026/osd-2026-campus-1.png" alt="Going through the talk slides" loading="lazy" />
      <div class="osd-gallery-caption">Going through the talk slides</div>
    </div>
    <div class="osd-gallery-item">
      <img src="/images/osd-2026/osd-2026-campus-2.png" alt="Slide: Teach in public" loading="lazy" />
      <div class="osd-gallery-caption">Slide: Teach in public</div>
    </div>
    <div class="osd-gallery-item">
      <video src="/images/osd-2026/osd-2026-peacock.mp4" autoplay muted loop playsinline></video>
      <div class="osd-gallery-caption">Peacock on campus, DAIICT Gandhinagar</div>
    </div>
  </div>
</div>

## What the event brought into focus

A lot of meaningful community work starts very close to the product. Someone tries to use the tool, notices rough edges, writes one fix, then one guide, then maybe a notebook, then eventually becomes the person welcoming others in. Contributors often come from users; advocates often come from contributors. The chain is shorter than people think.

That is part of why I still make time for events like this. The talk matters, but the surrounding conversations matter too, and attending these things keeps teaching me what to do, and what not to do, in the communities I help organize or volunteer with, including [SciPy India](https://scipy-india.github.io/).

## A few honest notes

The day had more roughness than I would have liked, and a fair bit of that came from the keynote block at the start. My talk began roughly an hour later than scheduled, largely because the back-to-back keynotes were not managed tightly enough on time. By the time the community track resumed, the room had already thinned out; once volunteers announced that workshops were running in parallel elsewhere, that only became more noticeable. It made the whole track feel less held together than it should have.

The keynotes themselves also did not leave a great impression on me. The first kept switching languages in a way that made it harder to follow. The second was harder to sit through: some unnecessary profanity, a few moments where attempts at audience engagement turned into cutting people off rather abruptly, and references to Bitcoin/cryptocurrency that did not really go anywhere. It all sat oddly in an event that had already circulated a code of conduct and speaker code of conduct to speakers beforehand.

That part stayed with me more than I expected. Having the CoC circulated to speakers ahead of time and then watching the keynotes go the way they did felt like a gap that was hard to ignore in practice.

Part of why I pay attention to these things is that every event I attend feeds back into the community organizing work I do elsewhere, including [SciPy India](https://scipy-india.github.io/). Good practices are worth borrowing; the weaker ones teach you something too.

## Heading back

On the way out, there was another peacock near the exit, this time properly fanned out. A proper show, really, and a decent note to leave on.

I will cross-post this on the FossUnited Discourse as well (and link to the thread here).

---
title: "Technical writing and community at Wikimedia DSDP"
description: "Three days with Wikimedia contributors in Delhi, a workshop on technical writing, and some thoughts on open knowledge."
pubDatetime: 2026-04-04T00:00:00Z
author: "Srihari Thyagarajan"
tags: ["wikimedia", "open-source", "community", "technical-writing", "conference", "workshop"]
featured: false
draft: false
---

Last weekend, I was in Delhi for the in-person conclusion of the [WikiMedia Developer Skill Development Program India](https://meta.wikimedia.org/wiki/Event:Indic_MediaWiki_Developers_User_Group/Developer_Skill_Development_Program_India_2025). [Krishna Chaitanya](https://www.linkedin.com/in/kcvelaga/) (everyone calls him KC, and I hopped on that pretty quick) had invited me and proposed I give a workshop; I suggested the topic of technical writing. The event was a three-day gathering of contributors who had spent the past four to six months building bots, gadgets, extensions, and web tools for the Wiki platform. Some of us were there as speakers, some were there to present what they had built, and some were both.

The [Indic MediaWiki Developers User Group](https://meta.wikimedia.org/wiki/Indic_MediaWiki_Developers_User_Group) organized the program, and this event served as its conclusion: contributors showed their work, got feedback, attended workshops, and met the people they had been collaborating with online.

## Friday: project showcases

The first day belonged to the contributors. Groups presented the projects they had been working on throughout the program, covering everything from MediaWiki extensions to bots and gadgets built for various Indic-language wiki communities. I sat in the audience for most of these, asked questions where I could, and got a sense of how broad the work had been. No two projects looked alike, and every group was building for a different community. You can explore the [full list of projects and their links here](https://meta.wikimedia.org/wiki/Event:Indic_MediaWiki_Developers_User_Group/Developer_Skill_Development_Program_India_2025#Projects).

A few talks accompanied the showcases as well, rounding out the day.

## Saturday: sessions and a social evening

Saturday was packed. The morning kicked off with a workshop by [Sapni G K](https://www.linkedin.com/in/sapni-g-k/): "More than Privacy Policies: Tech Development and the Evolving Regulatory Landscape." She walked through the intersection of tech development and regulation, covering data governance, governmental policies, and the ways those constraints show up in real product decisions. The session ended with a group activity that split us into policy makers vs. techies to debate tech policy from both sides. It was a good session to start with for a room full of people building tools for online communities.

[Aakash Shukla](https://www.linkedin.com/in/akashshkl) gave a talk on India's Digital Public Infrastructure and Open Source that I liked a lot. He had a dense slide deck and covered it without losing the room, which is hard to do. His background in startup mentorship and community building (Techstars, Google for Startups, UN, among others) came through in how he connected [Digital Public Goods for DPI](https://www.digitalpublicgoods.net/collections/coll-dpi) initiatives to the broader open source picture. He was also someone I kept running into and having good conversations with throughout the event.

[Sudhanshu Gautam](https://meta.wikimedia.org/wiki/User:SGautam_(WMF)), a designer at Wikimedia, ran a session on "UX Design 101 for Developers." He drew an analogy I liked: engineers have established patterns and principles they follow when writing code, and design has its own equivalent. Companies set branding guidelines, theming standards, color systems, and designers should treat those with the same rigor engineers treat their codebases. He referenced [Apple's 1987 Human Interface Guidelines](https://blog.prototypr.io/rediscovering-apples-human-interface-guidelines-1987-59731376b39e) as an early example of this kind of thinking done well. It's the kind of session that makes you look at the other side of a product you normally only touch as a developer.

It was also nice to see [Ashlesh Biradar](https://www.linkedin.com/in/ashleshbiradar) represent [FOSSUnited](https://fossunited.org/) with his session "Forking Around and Finding Out @ FOSS United." He walked through various initiatives they run, including financial support programs for open source in India. I have a soft spot for FOSSUnited since the community I help co-organize, [SciPy India](https://scipy-india.github.io/), was incubated at IndiaFOSS's [FOSS In Science devroom](https://fossunited.org/indiafoss/2025/devrooms/science) last year.

The day wrapped with a social outing the organizers had put together for the evening: a visit to [Museo Camera](https://museocamera.in/), a camera and photography museum curated by [Aditya Arya](https://www.adityaarya.com/). The album embedded below has images from the event overall, including some from the museum visit.

<style>
.wikimedia-gallery {
  position: relative;
  margin: 1.5rem 0;
}
.wikimedia-gallery-scroll {
  display: flex;
  gap: 1rem;
  overflow-x: auto;
  scroll-snap-type: x mandatory;
  scroll-behavior: smooth;
  -webkit-overflow-scrolling: touch;
  padding-bottom: 0.75rem;
  scrollbar-width: thin;
}
.wikimedia-gallery-scroll::-webkit-scrollbar {
  height: 6px;
}
.wikimedia-gallery-scroll::-webkit-scrollbar-track {
  background: transparent;
}
.wikimedia-gallery-scroll::-webkit-scrollbar-thumb {
  background: #888;
  border-radius: 3px;
}
.wikimedia-gallery-item {
  flex: 0 0 min(85%, 520px);
  scroll-snap-align: center;
  border-radius: 8px;
  overflow: hidden;
  position: relative;
  background: #111;
}
.wikimedia-gallery-item img {
  width: 100%;
  height: 360px;
  object-fit: cover;
  display: block;
  transition: transform 0.3s ease;
}
.wikimedia-gallery-item:hover img {
  transform: scale(1.02);
}
.wikimedia-gallery-item figcaption {
  padding: 0.5rem 0.75rem;
  font-size: 0.85rem;
  color: #ccc;
  text-align: center;
  background: #111;
}
.wikimedia-gallery-hint {
  text-align: center;
  font-size: 0.8rem;
  color: #888;
  margin-top: 0.25rem;
}
</style>

<div class="wikimedia-gallery">
  <div class="wikimedia-gallery-scroll">
    <figure class="wikimedia-gallery-item">
      <img src="/images/wikimedia-dsdp-2025/wikimedia-presentation.jpeg" alt="Wikimedia-branded session" loading="lazy" />
      <figcaption>Wikimedia-branded session</figcaption>
    </figure>
    <figure class="wikimedia-gallery-item">
      <img src="/images/wikimedia-dsdp-2025/Indic-MediaWiki-Developers-User-Group.jpeg" alt="Community chapter presentation" loading="lazy" />
      <figcaption>Community chapter presentation</figcaption>
    </figure>
    <figure class="wikimedia-gallery-item">
      <img src="/images/wikimedia-dsdp-2025/talk-privacy-policy-regulations.jpeg" alt="Sapni G K on tech regulation" loading="lazy" />
      <figcaption>Sapni G K on tech regulation</figcaption>
    </figure>
    <figure class="wikimedia-gallery-item">
      <img src="/images/wikimedia-dsdp-2025/talk-developer-responsibility.jpeg" alt="The developer responsibility" loading="lazy" />
      <figcaption>The developer responsibility</figcaption>
    </figure>
    <figure class="wikimedia-gallery-item">
      <img src="/images/wikimedia-dsdp-2025/talk-digital-public-infrastructure.jpeg" alt="Aakash Shukla on DPI and open source" loading="lazy" />
      <figcaption>Aakash Shukla on DPI and open source</figcaption>
    </figure>
    <figure class="wikimedia-gallery-item">
      <img src="/images/wikimedia-dsdp-2025/talk-open-protocol-specification.jpeg" alt="Open protocol specification" loading="lazy" />
      <figcaption>Open protocol specification</figcaption>
    </figure>
    <figure class="wikimedia-gallery-item">
      <img src="/images/wikimedia-dsdp-2025/talk-open-source-benefits.jpeg" alt="Open source contribution benefits" loading="lazy" />
      <figcaption>Open source contribution benefits</figcaption>
    </figure>
    <figure class="wikimedia-gallery-item">
      <img src="/images/wikimedia-dsdp-2025/workshop-session-wide-view.jpeg" alt="Workshop session, wide view" loading="lazy" />
      <figcaption>Workshop session, wide view</figcaption>
    </figure>
    <figure class="wikimedia-gallery-item">
      <img src="/images/wikimedia-dsdp-2025/project-demo.jpeg" alt="Workshop code demo" loading="lazy" />
      <figcaption>Workshop code demo</figcaption>
    </figure>
    <figure class="wikimedia-gallery-item">
      <img src="/images/wikimedia-dsdp-2025/writing-is-thinking-talk.jpeg" alt="Writing is thinking" loading="lazy" />
      <figcaption>Writing is thinking</figcaption>
    </figure>
    <figure class="wikimedia-gallery-item">
      <img src="/images/wikimedia-dsdp-2025/project-docs-ecosystem-talk.jpeg" alt="Your project docs are part of the ecosystem" loading="lazy" />
      <figcaption>Your project docs are part of the ecosystem</figcaption>
    </figure>
    <figure class="wikimedia-gallery-item">
      <img src="/images/wikimedia-dsdp-2025/code-is-cheap-show-me-the-talk.jpeg" alt="Code is cheap, show me the talk" loading="lazy" />
      <figcaption>Code is cheap, show me the talk</figcaption>
    </figure>
    <figure class="wikimedia-gallery-item">
      <img src="/images/wikimedia-dsdp-2025/museo-camera-building-exterior.jpeg" alt="Museo Camera, Gurgaon" loading="lazy" />
      <figcaption>Museo Camera, Gurgaon</figcaption>
    </figure>
    <figure class="wikimedia-gallery-item">
      <img src="/images/wikimedia-dsdp-2025/museo-camera-entrance-exhibit.jpeg" alt="The entrance exhibit" loading="lazy" />
      <figcaption>The entrance exhibit</figcaption>
    </figure>
    <figure class="wikimedia-gallery-item">
      <img src="/images/wikimedia-dsdp-2025/museo-camera-history-timeline.jpeg" alt="Photography history timeline" loading="lazy" />
      <figcaption>Photography history timeline</figcaption>
    </figure>
    <figure class="wikimedia-gallery-item">
      <img src="/images/wikimedia-dsdp-2025/museo-camera-guided-tour.jpeg" alt="Guided tour inside Museo Camera" loading="lazy" />
      <figcaption>Guided tour inside Museo Camera</figcaption>
    </figure>
    <figure class="wikimedia-gallery-item">
      <img src="/images/wikimedia-dsdp-2025/museo-camera-curator-interaction.jpeg" alt="With the curator" loading="lazy" />
      <figcaption>With the curator</figcaption>
    </figure>
    <figure class="wikimedia-gallery-item">
      <img src="/images/wikimedia-dsdp-2025/museo-camera-group-photo.jpeg" alt="Group photo at Museo Camera" loading="lazy" />
      <figcaption>Group photo at Museo Camera</figcaption>
    </figure>
  </div>
  <p class="wikimedia-gallery-hint">← scroll to see more →</p>
</div>

## Sunday: the workshop

Sunday was the day I had been looking forward to. My session, "Technical Writing in the Age of AI," was the first slot of the morning and ran for about ~90 minutes. Around 25 participants, most of them intermediate-to-advanced Wikimedia contributors ranging from students to working professionals.

The [slides](https://haleshot.github.io/talks/technical-writing-wikimedia-03-2026/) covered five sections: why the stakes around writing have shifted now that generation is cheap, what writing as a practice involves (drawing on frameworks from [Prashanth Rao](https://thedataquarry.com/blog/a-framework-for-reading-and-writing/) and [Shreya Shankar](https://www.sh-reya.com/blog/ai-writing/)), how to design documentation for real users using the [Diataxis framework](https://diataxis.fr/) with Wikimedia-specific examples, how to grow a writing practice over time, and how technical writing connects to community advocacy and devrel.

That last section was personal. My own path went something like this: OSS documentation PRs (for [marimo](https://marimo.io/), [CocoIndex](https://cocoindex.io/)) led to community recognition, which led to holdings booths at OSS events, which led to meeting KC, which led to this invitation. A few people I’d been speaking with were curious about what a technical writing path can look like in practice, so I used that part of the workshop to share some of my own trajectory.

The rest of Sunday's sessions wrapped up the program. The organizers collected feedback at the end since this was the first time they had run both the program and an in-person conclusion event. Good way to go about it, I think.

## Community, again

Community kept coming up throughout the three days. In the talks, in the project showcases, in conversations between sessions. I noticed it more than usual, probably because [this weekend I'm giving a talk at Open Source Weekend Gujarat](https://opensourceweekend.org/) on lessons learnt from building a community around a devtool. Spending three days with Wikimedia contributors right before that talk put a lot of things in perspective.

I also left Delhi wanting to tell more people in my network about Wikimedia as a place to contribute. Students and professionals, anyone with time and some (technical) interest. Wikimedia existed as one of the most important open knowledge resources long before LLMs started scraping it for training data, and it still is. Contributing to it doesn't need an incentive program attached; it's one of those spaces where the work is the reward, and you end up meeting like-minded people along the way, which is... the point. I'd encourage folks to look at it as something worth doing alongside (or instead of) the usual GSoC, Outreachy, and similar programs that tend to dominate the "how do I get into open source" conversation. (Yes, I know Wikimedia is also a part of GSoC. The point stands.)

## Heading back

Delhi weather and I did not get along; that part I could have done without. But the event was worth the trip. Thanks to Krishna Chaitanya and the Indic MediaWiki Developers User Group for putting this together and for the invitation. For a first iteration, they set a strong bar.

Oh, and between sessions the organizers would play [Listen to Wikipedia](https://listen.hatnote.com/) as background music. It sonifies real-time edits happening across Wikipedia; each sound corresponds to someone making an edit somewhere on a wiki page. Worked surprisingly well as elevator music between talks. Worth a listen if you haven't heard it.

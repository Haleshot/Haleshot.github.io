---
title: "Reflections on GitHub Constellation 2026"
description: "Notes from GitHub Constellation Bangalore: keynote takeaways, hallway conversations, GitHub's AI-era scale, and..."
pubDatetime: 2026-04-17T00:00:00Z
author: "Srihari Thyagarajan"
tags: ["github", "open-source", "community", "conference", "ai"]
featured: false
draft: false
---

Last weekend I was in Bangalore for [GitHub Constellation](https://githubconstellation.com/). It sits a bit outside my usual OSS conference orbit, but GitHub is one of the nicer company-organized ones; I wasn't going to skip it.

The [keynote](https://www.youtube.com/live/QNbJoEx36jA?si=ZAJRx9ELfIXaGUke) covered the usual ground: AI, agents, a revamped GitHub landing page with agentic features, cool demos. I'll leave most of the demos out; once you start recapping those it reads like a promo post. One exception: [Karan M V](https://www.linkedin.com/in/mvkaran/), GitHub's Director of DevRel, had some cool live coding demos during the [keynote](https://www.youtube.com/live/QNbJoEx36jA?si=ZAJRx9ELfIXaGUke); Copilot, GitHub Cloud, CLI, all synced up through the SDK. There was also an early preview of the new GitHub landing page, agentic by design, with features built around that (to be more native to the platform). [Kyle Daigle](https://github.com/kdaigle), GitHub's COO, also walked through the scale GitHub has had to absorb over the past year from the AI and agents influx, and [tweeted about some of it](https://x.com/kdaigle/status/2040164759836778878?s=20) too. The numbers were staggering. I'd love a proper engineering write-up from their team on this at some point. One detail I didn't expect: their bottleneck is CPUs, not GPUs. Kyle joked about the irony of that in the AI era, which got a laugh from the room.

I skipped most sessions after the keynotes. Once you've attended enough of these, you get selective about where you actually sit. The [full sessions list is here](https://githubconstellation.com/sessions) if you want to look. One I had been looking forward to was the Sarvam session, but Karan had invited me to an AMA with Kyle running at the same time, so that took priority.

<style>
.gh-constellation-gallery {
  position: relative;
  margin: 1.5rem 0;
}
.gh-constellation-gallery-scroll {
  display: flex;
  gap: 1rem;
  overflow-x: auto;
  scroll-snap-type: x mandatory;
  scroll-behavior: smooth;
  -webkit-overflow-scrolling: touch;
  padding-bottom: 0.75rem;
  scrollbar-width: thin;
}
.gh-constellation-gallery-scroll::-webkit-scrollbar {
  height: 6px;
}
.gh-constellation-gallery-scroll::-webkit-scrollbar-track {
  background: transparent;
}
.gh-constellation-gallery-scroll::-webkit-scrollbar-thumb {
  background: #888;
  border-radius: 3px;
}
.gh-constellation-gallery-item {
  flex: 0 0 min(85%, 520px);
  scroll-snap-align: center;
  border-radius: 8px;
  overflow: hidden;
  position: relative;
  background: #111;
}
.gh-constellation-gallery-item img {
  width: 100%;
  height: 360px;
  object-fit: cover;
  display: block;
  transition: transform 0.3s ease;
}
.gh-constellation-gallery-item:hover img {
  transform: scale(1.02);
}
.gh-constellation-gallery-item figcaption {
  padding: 0.5rem 0.75rem;
  font-size: 0.85rem;
  color: #ccc;
  text-align: center;
  background: #111;
}
.gh-constellation-gallery-hint {
  text-align: center;
  font-size: 0.8rem;
  color: #888;
  margin-top: 0.25rem;
}
</style>

<div class="gh-constellation-gallery">
  <div class="gh-constellation-gallery-scroll">
    <figure class="gh-constellation-gallery-item">
      <img src="/images/github-constellation-2026/india-oss-stats.jpeg" alt="Keynote slide showing India's open source AI contribution stats" loading="lazy" />
      <figcaption>India leads globally in open source contributors and is second only to the US for open source AI, with 7.5M+ contributions on GitHub</figcaption>
    </figure>
    <figure class="gh-constellation-gallery-item">
      <img src="/images/github-constellation-2026/ama-question.jpeg" alt="Asking Kyle Daigle about AI-generated PR spam at the AMA" loading="lazy" />
      <figcaption>Asking Kyle about the AI-generated PR problem at the AMA</figcaption>
    </figure>
    <figure class="gh-constellation-gallery-item">
      <img src="/images/github-constellation-2026/keynote-hall.jpeg" alt="Packed keynote hall at GitHub Constellation Bangalore" loading="lazy" />
      <figcaption>Packed keynote hall, GitHub Constellation Bangalore</figcaption>
    </figure>
    <figure class="gh-constellation-gallery-item">
      <img src="/images/github-constellation-2026/venue-foyer.jpeg" alt="Constellation '26 foyer with branded booths and Octocat" loading="lazy" />
      <figcaption>Constellation '26 foyer, Bangalore</figcaption>
    </figure>
    <figure class="gh-constellation-gallery-item">
      <img src="/images/github-constellation-2026/opening-keynote.jpeg" alt="Opening screens at the Constellation '26 keynote" loading="lazy" />
      <figcaption>Opening screens at the Constellation '26 keynote</figcaption>
    </figure>
    <figure class="gh-constellation-gallery-item">
      <img src="/images/github-constellation-2026/session.jpeg" alt="Session in progress at Constellation '26" loading="lazy" />
      <figcaption>Session in progress, Constellation '26</figcaption>
    </figure>
  </div>
  <p class="gh-constellation-gallery-hint">← scroll to see more →</p>
</div>

## The AMA

There was an AMA with Kyle (separate from the one listed on the [sessions page](https://githubconstellation.com/sessions)), and I got to ask a question. I gave Kyle some context before asking: there's a [thread in GitHub's community discussions](https://github.com/orgs/community/discussions/185387) that's been gaining traction around AI-generated PRs overwhelming open source maintainers. I brought up the [matplotlib incident](https://theshamblog.com/an-ai-agent-published-a-hit-piece-on-me/) too: an AI agent whose PR got rejected responded by publishing a reputational attack piece on the maintainer who closed it. The HuggingFace CEO had [tweeted](https://x.com/ClementDelangue/status/2034294644800974908?s=20) about their Transformers repo being flooded with agentic contributions from bots. And I mentioned a rumour circulating at the time: that [GitHub was looking at PR restrictions](https://www.infoworld.com/article/4127156/github-eyes-restrictions-on-pull-requests-to-rein-in-ai-based-code-deluge-on-maintainers.html) to rein in the volume, [reported in a few places](https://www.opensourceforu.com/2026/02/github-weighs-pull-request-kill-switch-as-ai-slop-floods-open-source/).

I won't quote Kyle's response. What came back was something close to: nobody has this figured out yet. Not maintainers, not individuals, not GitHub. These are norms and workflows that have been around long enough to feel settled, and AI is making everyone rethink them from scratch: maintainers, individuals, companies at GitHub's scale. Nobody has a clean playbook for that yet, and the answer didn't pretend otherwise.

I ran into [Akash Shukla](https://www.linkedin.com/in/akashshkl) at the event too (you might remember him from [my Wikimedia post](/posts/wikimedia-dsdp-experience/)). He introduced me to a bunch of people, several of whom had apparently already heard about me from him beforehand (he'd been sharing some of my recent writing around); funny, and it made those conversations much easier to start. I also met [Abhishek Mishra](https://www.linkedin.com/in/stalwartcoder/), devrel at [smallest.ai](https://smallest.ai/), and we had a good chat.

You can run a Discord or Zulip thread for months and still not get the same momentum as one good in-person conversation, and this keeps being true across every event. [Wrote about this in the PyConf Hyderabad writeup too](https://haleshot.github.io/posts/pyconf-hyd-experience/#why-being-there-helped).

The event was nicely put together. The hallways got congested when everyone spilled out between sessions; navigating between spaces was more of an effort than it needed to be. Minor complaint for an otherwise good day out.

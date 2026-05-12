---
title: "A day at ChennaiFOSS 2026"
description: "A day at IIT Madras with SciPy India, a few sharp sessions, and a reminder that local FOSS meetups still create momentum much faster than the internet does."
pubDatetime: 2026-05-13T16:06:08Z
author: "Srihari Thyagarajan"
tags: ["open-source", "community", "conference", "scipy-india", "foss", "chennai"]
featured: false
draft: false
---

I'm getting to this blog later than I care to admit. ChennaiFOSS 2026 happened on April 18, 2026, at IIT Madras in the [IC&SR building](https://icandsr.iitm.ac.in/), and I spent most of that day at the [SciPy India](https://scipy-india.github.io/) booth.

It was also my first time on the IIT Madras campus. I parked near the guest area, took the shuttle in, and, on the way to the venue, saw deer, blackbucks, and monkeys. It reminded me a little of [visiting](https://haleshot.github.io/posts/osd-2026-experience/) the DAAICT campus in Gujarat earlier this year. Apparently this is just what some of my conference travel looks like now, and I am not complaining!!

On the walk over, I ran into [Bowrna Prabhakaran](https://www.linkedin.com/in/bowrna/), open-source contributor and former Outreachy intern for Apache Airflow. She had already managed to [post about ChennaiFOSS](https://www.linkedin.com/posts/bowrna_last-week-i-attended-and-volunteered-at-share-7453310437963694081-ok3d?utm_source=share&utm_medium=member_desktop&rcm=ACoAADSJzvgBkjBd85IWDyUWA6ttzq8B-NDq-Hs) and [write about it properly](https://medium.com/@BowrnaPrabhu/my-experience-attending-chennaifoss-2026-3831e0972478), which only made me more aware that I was going to be late getting this one out. Once I got to the building and picked up my badge, I headed to set up the booth. I also met [Vivekanad S](https://www.linkedin.com/in/vivekanad-s-2777b0127/), whose talk was in the morning slot and whom we had earlier [hosted](https://www.youtube.com/live/6SoLgkFVkIg?si=8BnP9OoHq3AOWhBm&t=5609) on one of our SciPy India community calls.

<style>
.chennaifoss-gallery {
  position: relative;
  margin: 1.5rem 0;
}
.chennaifoss-gallery-scroll {
  display: flex;
  gap: 1rem;
  overflow-x: auto;
  scroll-snap-type: x mandatory;
  scroll-behavior: smooth;
  -webkit-overflow-scrolling: touch;
  padding-bottom: 0.75rem;
  scrollbar-width: thin;
}
.chennaifoss-gallery-scroll::-webkit-scrollbar {
  height: 6px;
}
.chennaifoss-gallery-scroll::-webkit-scrollbar-track {
  background: transparent;
}
.chennaifoss-gallery-scroll::-webkit-scrollbar-thumb {
  background: #888;
  border-radius: 3px;
}
.chennaifoss-gallery-item {
  flex: 0 0 min(85%, 520px);
  scroll-snap-align: center;
  border-radius: 8px;
  overflow: hidden;
  position: relative;
  background: #111;
}
.chennaifoss-gallery-item img {
  width: 100%;
  height: 360px;
  object-fit: cover;
  display: block;
  transition: transform 0.3s ease;
}
.chennaifoss-gallery-item:hover img {
  transform: scale(1.02);
}
.chennaifoss-gallery-item figcaption {
  padding: 0.5rem 0.75rem;
  font-size: 0.85rem;
  color: #ccc;
  text-align: center;
  background: #111;
}
.chennaifoss-gallery-hint {
  text-align: center;
  font-size: 0.8rem;
  color: #888;
  margin-top: 0.25rem;
}
</style>

<div class="chennaifoss-gallery">
  <div class="chennaifoss-gallery-scroll">
    <figure class="chennaifoss-gallery-item">
      <img src="/images/chennaifoss-2026/scipy-india-booth.jpg" alt="Talking to an attendee at the SciPy India booth during ChennaiFOSS 2026" loading="lazy" />
      <figcaption>Most of my day went into conversations at the SciPy India booth</figcaption>
    </figure>
    <figure class="chennaifoss-gallery-item">
      <img src="/images/chennaifoss-2026/accessible-curricula-talk.jpeg" alt="Sai Rahul Poruri speaking about making college curricula accessible at ChennaiFOSS 2026" loading="lazy" />
      <figcaption>Sai Rahul Poruri's lightning talk on making college curricula more accessible</figcaption>
    </figure>
    <figure class="chennaifoss-gallery-item">
      <img src="/images/chennaifoss-2026/future-of-foss-panel.jpeg" alt="Panel discussion on the future of FOSS, software engineering, and technical education at ChennaiFOSS 2026" loading="lazy" />
      <figcaption>The panel on the future of FOSS, software engineering, and technical education</figcaption>
    </figure>
    <figure class="chennaifoss-gallery-item">
      <img src="/images/chennaifoss-2026/session-crowd.jpg" alt="Attendees gathered inside a session room at ChennaiFOSS 2026" loading="lazy" />
      <figcaption>One of the session rooms, packed out well before the talks were done</figcaption>
    </figure>
    <figure class="chennaifoss-gallery-item">
      <img src="/images/chennaifoss-2026/fossee-session.jpg" alt="A speaker presenting FOSSEE offerings on a projection screen during ChennaiFOSS 2026" loading="lazy" />
      <figcaption>A FOSSEE session later in the day, one of the few I managed to catch away from the booth</figcaption>
    </figure>
  </div>
  <p class="chennaifoss-gallery-hint">← scroll to see more →</p>
</div>

Most of my day was the booth. That has become a familiar pattern by now, whether at [PyConf Hyderabad 2026](/posts/pyconf-hyd-experience/) or at smaller community meetups such as the [SciPy India x BangPypers meetup from February](https://scipy-india.github.io/blogs.html?id=scipy-india-x-bangpypers-feb-2026). In Chennai, most conversations fell into two buckets: people who remembered the original SciPy India conference and wanted to know what was happening now, and people hearing about it for the first time and asking how to get involved.

At some point during the day, Lakshmanan P stopped by and [interviewed me at the booth](https://www.linkedin.com/posts/lakshmanan-p_chennaifoss26-chennaifoss26-moolakaraconf-activity-7452200700241125379-_3ah?utm_source=share&utm_medium=member_desktop&rcm=ACoAADSJzvgBkjBd85IWDyUWA6ttzq8B-NDq-Hs). He was doing that with a lot of booth folks and speakers through the event, which I appreciated. He also later [posted a broader ChennaiFOSS recap](https://www.linkedin.com/posts/lakshmanan-p_chennaifoss2026-chennaifoss26-fossunited-ugcPost-7457488535474954240-ENLy?utm_source=share&utm_medium=member_desktop&rcm=ACoAADSJzvgBkjBd85IWDyUWA6ttzq8B-NDq-Hs).

I also met [Sai Rahul Poruri](https://rahulporuri.in/) in the morning. He introduced me to a few people, gave me a short walkthrough of nearby parts of campus later in the day, and we ended up speaking about SciPy India, event formats, and a few community ideas we would like to try. I'll leave that there for now.

Because I was holding the booth down alone for most of the day, I only made it to a few sessions.

One that stayed with me was Sai's lightning talk on [making college curricula accessible](https://fossunited.org/c/chennai/2026/cfp/7hcv05injj). I had seen him talk about the idea earlier, so I was curious how he would compress it into a lightning talk. A lot of course material is still much harder to access and navigate than it needs to be. He has also written about the topic on the [FossUnited forum](https://forum.fossunited.org/t/making-college-curricula-accessible/6949).

I also caught ["Why Labs Fear Distillation and how Smaller AI Models Are Built in the Open-Source World"](https://fossunited.org/c/chennai/2026/cfp/2mp22pmsg6). I liked that it leaned on concrete anecdotes, screenshots, and relevant tweets from prominent AI people instead of drifting into the usual open-vs-closed AI shadowboxing.

The panel discussion on ["The future of FOSS, SWE and Technical Education"](https://fossunited.org/c/chennai/2026/cfp/e37v2viqo0) was probably the session I got the most out of late in the day. [Ansh Arora](https://ansharora.in/) moderated it well and was willing to push back or double down on things the panelists said instead of just moving to the next question. Bowrna brought useful perspective from her work in software and open source, [Kaustubh Misra](https://www.linkedin.com/in/kaustubh-misra/) had some of the responses I found most interesting, and Rahul helped keep the discussion from becoming too narrow or too predictable.

The day ended with an open feedback round, and at some point a joke let slip that it was also the lead organizer's birthday. After that, it became a bit of a running gag for people to start with wishes before getting to the actual feedback. That felt about right.

It was a good event in my own city, I finally got to see the IIT Madras campus, and I had the kind of conversations that make local community spaces worth showing up for. I keep coming back to the same point after events like this: async spaces help, but in-person meetups create momentum much faster.

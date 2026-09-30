---
title: "IndiaFOSS 2026, this time with a devroom to run"
description: "Running the Documentation and Technical Writing devroom at IndiaFOSS 2026, giving a talk in it, and two days at the SciPy India booth."
pubDatetime: 2026-09-30T16:00:00Z
author: "Srihari Thyagarajan"
tags: ["open-source", "community", "conference", "foss", "write-the-docs", "documentation", "scipy-india", "bengaluru"]
featured: false
draft: false
---

IndiaFOSS 2026 happened on 26 and 27 September at NIMHANS Convention Centre in Bengaluru, and it was a very different IndiaFOSS for me from the last time I attended :) Along with [Sujatha Mohan](https://www.linkedin.com/in/sujatharamakrishnan/) and [Agriya Khetarpal](https://agriyakhetarp.al/), I co-ran the [Documentation and Technical Writing devroom](https://fossunited.org/indiafoss/2026/devrooms/docs) on Saturday morning, the first thing [Write the Docs India](https://write-the-docs-india.github.io/) has done since we restarted the chapter. I spent most of the remaining time at the [SciPy India](https://scipy.in) booth.

<style>
.indiafoss-gallery {
  position: relative;
  margin: 1.5rem 0;
}
.indiafoss-gallery-scroll {
  display: flex;
  gap: 1rem;
  overflow-x: auto;
  scroll-snap-type: x mandatory;
  scroll-behavior: smooth;
  -webkit-overflow-scrolling: touch;
  padding-bottom: 0.75rem;
  scrollbar-width: thin;
}
.indiafoss-gallery-scroll::-webkit-scrollbar {
  height: 6px;
}
.indiafoss-gallery-scroll::-webkit-scrollbar-track {
  background: transparent;
}
.indiafoss-gallery-scroll::-webkit-scrollbar-thumb {
  background: #888;
  border-radius: 3px;
}
.indiafoss-gallery-item {
  flex: 0 0 min(85%, 520px);
  scroll-snap-align: center;
  border-radius: 8px;
  overflow: hidden;
  position: relative;
  background: #111;
}
.indiafoss-gallery-item img {
  width: 100%;
  height: 360px;
  object-fit: cover;
  display: block;
  transition: transform 0.3s ease;
}
.indiafoss-gallery-item img.indiafoss-contain {
  object-fit: contain;
}
.indiafoss-gallery-item:hover img {
  transform: scale(1.02);
}
.indiafoss-gallery-item figcaption {
  padding: 0.5rem 0.75rem;
  font-size: 0.85rem;
  color: #ccc;
  text-align: center;
  background: #111;
}
.indiafoss-gallery-hint {
  text-align: center;
  font-size: 0.8rem;
  color: #888;
  margin-top: 0.25rem;
}
</style>

## Friday

I missed Friday's [unconference](https://fossunited.org/c/indiafoss/maintainer-summit) and the [pre-event the Tangled folks ran](https://luma.com/by2j97ga), since I was working. Next time! (I did catch them at their booth the next day, and we got talking about their vouching system. I knew of [Vouch](https://github.com/mitchellh/vouch) from the Ghostty folks, and wanted to know if theirs drew on something like it. It turned into a great conversation.) We also got a little of the devroom setup done that evening, which made Saturday morning easier.

## Saturday morning

I was at the registration desk early. As devroom managers we had a list of things to sort before the first talk: loading every speaker's deck onto my laptop so the whole morning could run off one machine, and working through the livestream and A/V with the event volunteers. By the time people started coming in, things were in order (mostly; the livestream took longer to get going than we planned, and that ate the slot we'd set aside for an unconference).

## The devroom

A lot of people turned up!! At one point the room was overflowing. For something that started as a proposal I wasn't sure would get picked (a whole room just for docs?), that was a very nice thing to see.

I won't go through every talk here, since we wrote those up properly in the [devroom debrief on the Write the Docs India site](https://write-the-docs-india.github.io/debriefs/2026/indiafoss-2026-devroom/). The whole morning is also [on YouTube](https://www.youtube.com/live/KNKAi9dfZxA).

We got everyone together for a group photo partway through, just before my own lightning talk, [Refactoring documentation without breaking it](https://fossunited.org/c/indiafoss/2026/cfp/6dpd2lbkma). The short version: renaming a heading or moving a page is a refactor, but our editors treat it as a text edit, so links break and CI only tells you once the site is live. One of my suggestions was putting a link checker like [lychee](https://github.com/lycheeverse/lychee) in your docs CI. It felt great to give, and I finished on time, which mattered that morning: one of the volunteers was keeping time, and we were trying hard to stick to it to make up for the minutes lost at the start. The [slides are here](https://haleshot.github.io/talks/indiafoss-docs-refactoring-09-2026/) and the [recording starts at 1:18:38](https://www.youtube.com/live/KNKAi9dfZxA?t=4718).

The conversations before and after were my favourite part of the morning. I went around asking people for feedback, and more than a few told me they were glad someone was starting the chapter back up. Hearing that made the months of emails and planning feel very worth it.

<div class="indiafoss-gallery">
  <div class="indiafoss-gallery-scroll">
    <figure class="indiafoss-gallery-item">
      <img src="/images/indiafoss-2026/docs-devroom-audience.jpeg" alt="A full room of attendees seated at long desks in the Documentation and Technical Writing devroom at IndiaFOSS 2026" loading="lazy" />
      <figcaption>A full room for the docs devroom</figcaption>
    </figure>
    <figure class="indiafoss-gallery-item">
      <img src="/images/indiafoss-2026/ganesh-bruno-talk.jpeg" alt="Ganesh Patil presenting a slide titled What this talk is about during his talk on documenting Bruno" loading="lazy" />
      <figcaption>Ganesh Patil on two years of documenting Bruno</figcaption>
    </figure>
    <figure class="indiafoss-gallery-item">
      <img class="indiafoss-contain" src="/images/indiafoss-2026/my-talk-livestream.png" alt="A frame from the livestream showing Srihari speaking next to a slide titled If you maintain docs for a project" loading="lazy" />
      <figcaption>Giving my lightning talk, a few slides from the end (a frame from the livestream)</figcaption>
    </figure>
  </div>
  <p class="indiafoss-gallery-hint">← scroll to see more →</p>
</div>

## Around the conference

Between the devroom and the booth, I dropped into a few other rooms: the Open Design devroom for the [Forkable Design talk](https://fossunited.org/c/indiafoss/2026/cfp/2tsmo50knt), [Jaidev Deshpande's talk](https://fossunited.org/c/indiafoss/2026/cfp/dglc657qff), and a couple of others.

## The SciPy India booth

Saturday afternoon and all of Sunday went into the SciPy India booth with [Malayaja Chutani](https://www.linkedin.com/in/malayajachutani/), [Agriya](https://agriyakhetarp.al/), and [Aditi Juneja](https://www.linkedin.com/in/aditi-juneja-940838204/) (the rest of the [SciPy India team](https://scipy.in/2026/team)), and [Rahul Poruri](https://rahulporuri.in/) (FOSS United's CEO) dropping in every now and then.

It was a lot of conversations!! Most of them started with what SciPy India is these days and ended with the conference we're running at IIT Madras on 19 and 20 December, with me nudging people to [submit a talk or a workshop](https://cfp.scipy.in/scipy-india-2026) before the CFP closes on 19 October. I also managed to catch [Kailash Nadh](https://nadh.in/) during the conference and talk briefly about SciPy India.

<div class="indiafoss-gallery">
  <div class="indiafoss-gallery-scroll">
    <figure class="indiafoss-gallery-item">
      <img class="indiafoss-contain" src="/images/indiafoss-2026/scipy-booth-crowd.jpeg" alt="Srihari at the SciPy India booth with a group of attendees reading the handwritten signs on the table" loading="lazy" />
      <figcaption>The SciPy India booth during one of its busier stretches</figcaption>
    </figure>
    <figure class="indiafoss-gallery-item">
      <img src="/images/indiafoss-2026/scipy-booth-desk.jpeg" alt="Handwritten SciPy India signs on the booth table, including one announcing SciPy India 2026 at IIT Madras on 19 and 20 December with the CFP closing on 19 October" loading="lazy" />
      <figcaption>Our handwritten booth signs, CFP date included</figcaption>
    </figure>
    <figure class="indiafoss-gallery-item">
      <img src="/images/indiafoss-2026/scipy-booth-conversations.jpeg" alt="Attendees gathered around the SciPy India booth talking to the team" loading="lazy" />
      <figcaption>More conversations at the booth</figcaption>
    </figure>
    <figure class="indiafoss-gallery-item">
      <img src="/images/indiafoss-2026/scipy-stickers.jpeg" alt="A fan of SciPy India 2026 stickers laid out on a dark table" loading="lazy" />
      <figcaption>SciPy India 2026 stickers</figcaption>
    </figure>
    <figure class="indiafoss-gallery-item">
      <img src="/images/indiafoss-2026/scipy-booth-team.jpeg" alt="Agriya, Srihari, and Malayaja sitting at the SciPy India booth holding up handwritten signs" loading="lazy" />
      <figcaption>Agriya, me, and Malayaja at the booth</figcaption>
    </figure>
    <figure class="indiafoss-gallery-item">
      <img class="indiafoss-contain" src="/images/indiafoss-2026/united-by-foss-frame.jpeg" alt="Srihari, Malayaja, and Agriya standing inside the sticker-covered United by FOSS photo frame" loading="lazy" />
      <figcaption>Srihari (me), Agriya, and Malayaja in the United by FOSS frame</figcaption>
    </figure>
    <figure class="indiafoss-gallery-item">
      <img src="/images/indiafoss-2026/forkable-design-talk.jpeg" alt="The Forkable Design talk in the Open Design devroom, with the title slide on screen" loading="lazy" />
      <figcaption>The Forkable Design talk in the Open Design devroom</figcaption>
    </figure>
    <figure class="indiafoss-gallery-item">
      <img src="/images/indiafoss-2026/zasper-booth.jpeg" alt="Prasun Anand pointing at a screen while talking to attendees at the Zasper booth" loading="lazy" />
      <figcaption>Prasun Anand walking people through Zasper at its booth</figcaption>
    </figure>
    <figure class="indiafoss-gallery-item">
      <img src="/images/indiafoss-2026/foyer.jpeg" alt="Attendees in the NIMHANS Convention Centre foyer under a large IndiaFOSS 2026 banner" loading="lazy" />
      <figcaption>The foyer between sessions</figcaption>
    </figure>
  </div>
  <p class="indiafoss-gallery-hint">← scroll to see more →</p>
</div>

## Until next year

IndiaFOSS keeps getting better to look forward to. It's one of the largest FOSS conferences in India (if not THE biggest), and every year it's lots of familiar faces and plenty of new ones. It has become the weekend I set aside on my calendar well in advance, and this year it came with a devroom to run too!!

And for Write the Docs India, after a few quiet years, Saturday was the first time in a long while the chapter had a room full of people again. There's clearly a lot of interest, so city meetups are next. If you write or look after docs and want to help run one where you live, come find us on the [WhatsApp community](https://chat.whatsapp.com/EwspIErzCWQ4SI4MXKB4D0?mode=gi_t).

Thanks to Sujatha and Agriya for running this with me, to every speaker, to everyone who came, and to the FOSS United team and IndiaFOSS volunteers who made both days work.

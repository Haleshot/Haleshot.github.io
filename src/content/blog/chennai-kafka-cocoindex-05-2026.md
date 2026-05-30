---
title: "Co-organizing and speaking at the Chennai Kafka meetup"
description: "On co-organizing the Chennai Apache Kafka meetup, speaking about CocoIndex's Kafka connector, and a Flink session the room didn't want to end."
pubDatetime: 2026-05-30T12:00:00Z
author: "Srihari Thyagarajan"
tags: ["apache-kafka", "cocoindex", "open-source", "meetups", "chennai"]
featured: false
draft: false
---

I spoke at the [Apache Kafka Chennai meetup](https://www.meetup.com/chennai-kafka/) at [Facilio](https://maps.app.goo.gl/whLi6bB7P7U2MLBs9) this weekend. [Ena Koide](https://www.linkedin.com/in/ena-koide/), [Sai Krishna](https://www.linkedin.com/in/sk-21/), and I co-organized it. A few people have reached out wanting to volunteer, so the co-organizer list might be longer the next time I write about one of these!!

My talk was "Declare State, Not Messages". The idea was simple. Changing files and messy knowledge sources (docs, repos, wikis, PDFs) can become ordinary keyed Kafka records without making every downstream consumer rebuild its own integration logic. Instead of writing producer code that decides every send (send this row again, skip that one, emit a delete, re-emit everything after a restart), you describe what each key in the topic should contain, and CocoIndex emits only the upserts, tombstones, and no-ops needed to reconcile.

I kept the demo small. A few CSV rows changed locally, CocoIndex noticed the difference, and Kafka received only what changed. That was the point. You can find [my slides here](https://haleshot.github.io/talks/chennai-kafka-cocoindex-05-2026/).

<style>
.kafka-gallery {
  position: relative;
  margin: 1.5rem 0;
}
.kafka-gallery-scroll {
  display: flex;
  gap: 1rem;
  overflow-x: auto;
  scroll-snap-type: x mandatory;
  scroll-behavior: smooth;
  -webkit-overflow-scrolling: touch;
  padding-bottom: 0.75rem;
  scrollbar-width: thin;
}
.kafka-gallery-scroll::-webkit-scrollbar {
  height: 6px;
}
.kafka-gallery-scroll::-webkit-scrollbar-track {
  background: transparent;
}
.kafka-gallery-scroll::-webkit-scrollbar-thumb {
  background: #888;
  border-radius: 3px;
}
.kafka-gallery-item {
  flex: 0 0 min(85%, 520px);
  scroll-snap-align: center;
  border-radius: 8px;
  overflow: hidden;
  position: relative;
  background: #111;
}
.kafka-gallery-item img {
  width: 100%;
  height: 360px;
  object-fit: cover;
  display: block;
  transition: transform 0.3s ease;
}
.kafka-gallery-item:hover img {
  transform: scale(1.02);
}
.kafka-gallery-item figcaption {
  padding: 0.5rem 0.75rem;
  font-size: 0.85rem;
  color: #ccc;
  text-align: center;
  background: #111;
}
.kafka-gallery-hint {
  text-align: center;
  font-size: 0.8rem;
  color: #888;
  margin-top: 0.25rem;
}
</style>

<div class="kafka-gallery">
  <div class="kafka-gallery-scroll">
    <figure class="kafka-gallery-item">
      <img src="/images/chennai-kafka-cocoindex-05-2026/networking.jpeg" alt="Attendees standing and talking during a break between sessions at the Apache Kafka Chennai meetup" loading="lazy" />
      <figcaption>Catching up during a break between sessions</figcaption>
    </figure>
    <figure class="kafka-gallery-item">
      <img src="/images/chennai-kafka-cocoindex-05-2026/meetup-room.jpeg" alt="Mani Selvan presenting to a seated audience, a Kafka slide on screen" loading="lazy" />
      <figcaption>Mani on stream processing</figcaption>
    </figure>
    <figure class="kafka-gallery-item">
      <img src="/images/chennai-kafka-cocoindex-05-2026/session-slides.jpeg" alt="Mani Selvan presenting Kafka architecture slides during his Flink session" loading="lazy" />
      <figcaption>Mani walking through Kafka architecture</figcaption>
    </figure>
    <figure class="kafka-gallery-item">
      <img src="/images/chennai-kafka-cocoindex-05-2026/speaker-presenting.jpeg" alt="Srihari presenting the CocoIndex Kafka connector at the screen" loading="lazy" />
      <figcaption>Presenting the CocoIndex Kafka connector</figcaption>
    </figure>
    <figure class="kafka-gallery-item">
      <img src="/images/chennai-kafka-cocoindex-05-2026/session-in-progress.jpeg" alt="Srihari presenting the CocoIndex talk to a seated audience at the meetup" loading="lazy" />
      <figcaption>Declaring state, not messages</figcaption>
    </figure>
    <figure class="kafka-gallery-item">
      <img src="/images/chennai-kafka-cocoindex-05-2026/audience.jpeg" alt="The audience seated during Mani Selvan's Apache Flink session" loading="lazy" />
      <figcaption>The room during Mani's Flink session</figcaption>
    </figure>
  </div>
  <p class="kafka-gallery-hint">← scroll to see more →</p>
</div>

I proposed the talk because I contribute to [CocoIndex](https://github.com/cocoindex-io/cocoindex) and the project recently shipped a [Kafka connector](https://cocoindex.io/docs/connectors/kafka/). That made the meetup a good place to explain the connector through the problem it solves rather than through implementation details. My one-line version of the talk was to declare what each key should contain, let CocoIndex reconcile the difference, and let Kafka fan it out.

The other speaker was [Mani Selvan K](https://www.linkedin.com/in/mani-selvan-k-5820b692/), who spoke about stream processing fundamentals with Apache Flink; Flink SQL, Kafka integration, state management, and real-time pipeline design. The room took to it. (His session ran a little long on time, and we had to wrap fast to make it to lunch, so a few people wished he'd gotten to the last few Flink slides.) He also [posted a short note about the meetup](https://www.linkedin.com/posts/mani-selvan-k-5820b692_today-i-had-the-opportunity-to-speak-at-ugcPost-7466493490030542849-xjd8/) that captured the day well, and his [Flink 101 slides are on GitHub](https://github.com/ManiselvanSE/Flink101_Presentation/tree/main).

Presenting to a room that knows Kafka well meant the questions got specific fast, and that's most of what I come to these for.

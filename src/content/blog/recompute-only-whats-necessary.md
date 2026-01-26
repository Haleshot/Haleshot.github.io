---
title: "Déjà Vu: Recompute Only What's Necessary"
description: "On finding myself drawn to tools that embrace incremental computation."
pubDatetime: 2026-01-26T00:00:00Z
author: "Srihari Thyagarajan"
tags: ["devtools", "open-source", "philosophy"]
featured: true
draft: false
---

First post of 2026.

I've been noticing a pattern in the tools I gravitate toward; they all seem to share the same philosophy. *Recompute only what's necessary.* Don't redo work that doesn't need redoing. It showed up first with [marimo](https://marimo.io) and reactive notebooks; now it's showing up again with [CocoIndex](https://cocoindex.io) and incremental data processing. I think there's something to this.

There's another pattern too: I keep finding myself drawn to devtools and frameworks in their *early stages* of development. The kind where user feedback actually shapes the product, where maintainers are accessible, where your contributions can have real impact. That's the sweet spot.

## When the Framework Picks the Language

Here's something I find genuinely interesting: I always assumed you learn a language first, *then* find frameworks that use it. But for me it's been the opposite. marimo pulled me deeper into Python; CocoIndex is now pulling me into Rust. The project comes first, the language follows.

> I think this happens when a tool's design philosophy resonates with you so strongly that learning its underlying language becomes a natural byproduct rather than a prerequisite.

## OSS Contributions as a Way of Understanding

Contributing to open source deepens your understanding of a tool in ways that just *using* it never can. You start seeing the design decisions, the trade-offs, the places where the maintainers had to choose between competing priorities.

Reading the recent [Slides-to-Speech blog](https://cocoindex.io/blogs/slides-to-speech) is a good example. My mind immediately started mapping where else this pattern could apply. CFP platforms like [Papercall](https://papercall.io) and [Pretalx](https://pretalx.com) came to mind right away (maybe my mind is just wired this way).

Now exploring whether CocoIndex could work for these use cases. [Posted about this](https://x.com/hari_leo03/status/2014905695095910508) and started a discussion on the [CocoIndex Discord](https://discord.com/channels/1314801574169673738/1324891077693669438/1464870495253172407) if you want to help build a Pretalx plugin.

Earlier this month, gave a talk on CocoIndex at Rust Delhi; you can read about it [here](https://www.linkedin.com/posts/srihari-thyagarajan_kicking-off-2026-with-rust-delhi-meetup-12-activity-7420450968770764800-SQyb).

## The Philosophy That Keeps Showing Up

The analogy between marimo and CocoIndex feels almost too neat. marimo recomputes notebook cells reactively; only the cells affected by a change get re-executed. CocoIndex does the same thing for data pipelines; it processes _incrementally_, streaming only what's new or modified. Both reject the idea of brute-force recomputation. Both trust that most of the work you did before is still valid.

This "recompute only what's necessary" philosophy is underrated. It shows up in [React's reconciliation algorithm](https://legacy.reactjs.org/docs/reconciliation.html). It shows up in [artifact-based build systems](https://bazel.build/basics/artifact-based-builds). It shows up in reactive programming more broadly. Once you start noticing it, you see it everywhere.

## Looking Ahead

I'll be writing more this year; if not for engagement or feedback, then at least as an archive for myself. There's something valuable about documenting what you're working on, even if the audience is mostly future-you.

And if the pattern holds, I suspect I'll find myself drawn to yet another tool that shares this same philosophy. Looking forward to finding out what it is.

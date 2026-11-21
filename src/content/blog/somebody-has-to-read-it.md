---
title: "Somebody has to read it"
description: "I draft with a model, and a post about unreadable AI writing still got to me. Notes on what I owe whoever reads the result."
pubDatetime: 2026-11-16T00:00:00Z
author: "Srihari Thyagarajan"
tags: ["writing", "ai", "technical-writing", "opinion"]
featured: false
draft: false
---

In my [first post of this year](https://haleshot.github.io/posts/recompute-only-whats-necessary) I said I'd write more, "if not for engagement or feedback, then at least as an archive for myself." I've mostly kept to that, if event recaps count (I'm counting them).

This one's about something that's been bugging me for a few weeks.

In March I ran a workshop called "Technical Writing in the Age of AI" at the [Wikimedia DSDP event in Delhi](https://haleshot.github.io/posts/wikimedia-dsdp-experience) (the [slides are here](https://haleshot.github.io/talks/technical-writing-wikimedia-03-2026/)). One section was on what writing as a practice involves, and two posts I like came up a fair bit there: Shreya Shankar's [Writing in the age of LLMs](https://www.sh-reya.com/blog/ai-writing/) and Prashanth Rao's [framework for technical writing](https://thedataquarry.com/blog/a-framework-for-reading-and-writing/). Both of them use a model when they write, and both are careful about how.

So I'm in the "use the tools" camp; I even have [a page about it](https://haleshot.github.io/ai/), and this post was put together the same way. Then I read Colin Breck's [I don't want to read what you didn't write](https://blog.colinbreck.com/i-dont-want-to-read-what-you-didnt-write/), which is a long way from that camp, and caught myself nodding at paragraph after paragraph.

## The reading hasn't gotten any faster

Ask any senior engineer how their week went and there's a good chance the answer involves a [review queue that won't stop growing](https://blog.pragmaticengineer.com/the-pulse-new-trend-concern-about-massive-increase-in-code-review-load/) (agents opening PRs faster than anyone can look at them). Kailash Nadh wrote about this in January, in [Code is cheap. Show me the talk.](https://nadh.in/blog/code-is-cheap/) (I used the title on one of my workshop slides): "reading and critically evaluating code have become more important than learning syntax and typing it out line by line."

Docs aren't very different. Putting together a design doc or a PRD is a lot quicker now; sitting with one and understanding what it's asking of you is a whole different story.

<figure>
  <img src="/images/somebody-has-to-read-it/reading-pile.png" alt="Hand-drawn sketch on ruled paper: a design doc, a PRD and a blog draft, each written in a few minutes, land on a growing to-read pile; one person reads them one at a time" loading="lazy" />
  <figcaption>The pile grows faster than anyone reads it.</figcaption>
</figure>

The part of [Colin Breck's post](https://blog.colinbreck.com/i-dont-want-to-read-what-you-didnt-write/) I agree with most is about context. When I'm the one prompting, I know what I asked for, what I threw away, and which files the model looked at; the output reads fine *to me*. Whoever opens it next has none of that. He describes that reader as "forced to read exhaustively and consider every line, peering into the internals of a machine", which is how it feels from the receiving end.

<!-- Srihari: I couldn't fill this from your site. If you have a real instance (a doc, PRD or proposal that reached you and was plainly generated, and what you did with it), give me two lines and I'll write it in. Otherwise delete this comment. -->

## I still draft with help

Murat Demirbas's [Writing as a thinking tool](https://muratbuffalo.blogspot.com/2026/11/writing-as-thinking-tool.html) makes the case for keeping the model out of it until the thinking's done. For him the document is almost a *by-product*; the thinking happens in the writing, and a model can't do that part for you. He quotes Leslie Lamport:

> If you think without writing, you only think you're thinking.

(That same slide says "Writing is thinking". Lamport's version is better.)

I'm with Murat on all of that, and I still open a chat window earlier than he'd like. Shreya's [post](https://www.sh-reya.com/blog/ai-writing/) (the one from the workshop) has a version of this I like; she says she's "usually sharper at critiquing than generating", so she writes each paragraph herself first, "even if it’s ugly", and brings a model in when she's stuck on how to say something she already knows she wants to say. That's close to how I work!! I'd rather get a rough version down fast and spend my energy on the edit, where I'm more useful anyway.

Prashanth's [framework post](https://thedataquarry.com/blog/a-framework-for-reading-and-writing/), the other one from the workshop, kept the model at arm's length: ideas and outline by hand, a model only for polish at the end. The one he wrote a few months later, on [how he writes faster with AI](https://thedataquarry.com/blog/how-i-5x-my-writing-throughput-with-ai/), has a model at every stage, drafting included. What I like is what he *kept*. He still checks the outline himself before any drafting starts, and ownership of the ideas and the narrative is, in his word, "non-negotiable". That's roughly where I've landed too.

## A prompt's worth

Jacob Tomlinson gave me the vocabulary for doing this well versus badly, in [a post he dictated while walking his dog](https://jacobtomlinson.dev/posts/2026/writing-blog-posts-while-walking-the-dog). He calls the bad kind *extrapolation*: a couple of lines go in, two thousand words come out, and the reader gets to hunt for the couple of lines.

> You might as well just tweet the prompt instead of blowing it up into a piece of content.

The other kind is *interpolation*. You turn up with more than will fit (saved links, notes, slides, half an argument) and the model helps you cut it down.

<figure>
  <img src="/images/somebody-has-to-read-it/extrapolation-vs-interpolation.png" alt="Hand-drawn sketch on graph paper: in extrapolation a prompt is inflated into a long page where the prompt is one highlighted line; in interpolation a pile of notes, links and slides is boiled down into a short post that is highlighted throughout" loading="lazy" />
  <figcaption>Jacob's two modes, from the reader's side.</figcaption>
</figure>

That's also my answer to the context problem from [Colin's post](https://blog.colinbreck.com/i-dont-want-to-read-what-you-didnt-write/), I think. If the pile was *mine*, what lands in front of you is a shorter version of things I know and can defend. If all I had was the prompt, you're reading the model's guess at what I meant.

## My name's on it

Going back to [Kailash Nadh's post](https://nadh.in/blog/code-is-cheap/) from earlier; a good chunk of it is about accountability, and that's the part I'd underline. Endless generated output is hard to value, he argues, and what gives a piece of work its worth is

> the ability to hold someone accountable at least emotionally and morally, and sometimes legally, for an artefact

No model can take that off my hands. It can draft a sentence; it can't be the person who gets the "wait, is this right?" reply. And if rerunning the model is all I add, [there isn't much left for me to answer for](https://danluu.com/brain-off/). The rule on my AI page says it shorter: "if I can’t explain it in my own words, I don’t write about it."

<figure>
  <img src="/images/somebody-has-to-read-it/who-answers-for-it.png" alt="Hand-drawn sketch on dotted paper: my pile, my outline, a first pass, my edit, then a published and signed page; the reader's question comes back to me" loading="lazy" />
  <figcaption>Whatever happens in the middle, the reply comes back to me.</figcaption>
</figure>

The same rule applies to picking up anything new; I'd much rather a model [quiz me than hand me the answer](https://ogzhanolguncu.com/blog/how-to-keep-learning-in-the-age-of-llms/).

## Keeping the rough edges

Colin Breck ends [I don't want to read what you didn't write](https://blog.colinbreck.com/i-dont-want-to-read-what-you-didnt-write/) with this:

> I don’t want to live in a world where you use AI to summarize something important into unreadable text, and then I use AI in an attempt to decipher it. I want to hear you, imperfections and all. I want your interpretation of aesthetics, beauty, quality, relationship, time. I want to know how you feel. I want you to cut through and tell me what really matters. I don’t want to read a summary of facts and connections separate from experience and relationship.

I read a bit of *cringe* into it, even though he never uses the word. A lot of writing gets smoothed over because someone's worried about sounding too eager, or too plain, or about a joke that won't land. However a post gets written, I think those are the bits worth leaving in.

Also, if you're anywhere near Chennai on 19 and 20 December: we're running [SciPy India](https://scipy.in/2026/) at IIT Madras (workshops on the Saturday, talks on the Sunday), and [tickets are up](https://konfhub.com/scipy-india-2026). Would love to see you there :)

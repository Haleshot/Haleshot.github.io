// Talks I've given. Newest first.
// Titles/venues mirror https://haleshot.github.io/talks/ (source: github.com/Haleshot/talks).
// Each entry needs at least one URL because the homepage renders talks as links.
type TalkBase = {
  title: string;
  venue: string;
  date: string; // human-readable, e.g. "May 2026"
};

type TalkWithSlides = TalkBase & {
  slides: string;
  post?: string; // related blog post on this site
};

type TalkWithPost = TalkBase & {
  slides?: string;
  post: string;
};

export type Talk = TalkWithSlides | TalkWithPost;

export const TALKS: Talk[] = [
  {
    title: "Declare State, Not Messages",
    venue: "Apache Kafka Meetup, Chennai",
    date: "May 2026",
    slides: "https://haleshot.github.io/talks/chennai-kafka-cocoindex-05-2026/",
    post: "/posts/2026/chennai-kafka-cocoindex-05-2026",
  },
  {
    title: "Lessons from Building a Devtool Community",
    venue: "Open Source Weekend, Gandhinagar",
    date: "April 2026",
    slides: "https://haleshot.github.io/talks/open-source-weekend-04-2026/",
    post: "/posts/2026/osd-2026-experience",
  },
  {
    title: "Technical Writing in the Age of AI",
    venue: "Wikimedia DSDP",
    date: "March 2026",
    slides: "https://haleshot.github.io/talks/technical-writing-wikimedia-03-2026/",
    post: "/posts/2026/wikimedia-dsdp-experience",
  },
  {
    title: "Stop reprocessing everything: incremental data pipelines with CocoIndex",
    venue: "Rust Delhi Meetup",
    date: "January 2026",
    slides: "https://haleshot.github.io/talks/rust-delhi-cocoindex-01-2026/",
  },
];

import type { CollectionEntry } from "astro:content";

// Event recaps are posts tagged as a conference or meetup write-up
export const isRecap = (post: CollectionEntry<"blog">) =>
  post.data.tags.some((tag) => tag === "conference" || tag === "meetups");

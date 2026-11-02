import { BYLINES, bylineSource, outletIcon } from "@/data/bylines";
import { TALKS } from "@/data/talks";

export type Related = {
  label: string;
  title: string;
  href: string;
  icon?: string;
  source?: string; // byline outlet key, so the posts page filter can match the parent post
  recording?: string;
};

// Posts live at both /posts/<slug> and /posts/<year>/<slug>, so match on the slug
const slugOf = (path: string) => path.replace(/\/$/, "").split("/").pop();

// Talks and outside write-ups that belong to one of my posts
export const getRelated = (postPath: string): Related[] => [
  ...TALKS.filter((t) => t.post && slugOf(t.post) === slugOf(postPath) && t.slides).map((t) => ({
    label: "Talk",
    title: t.title,
    href: t.slides as string,
    recording: t.recording,
  })),
  ...BYLINES.filter((b) => b.related && slugOf(b.related) === slugOf(postPath)).map((b) => ({
    label: b.outlet.replace(/ blog$/, ""),
    title: b.title,
    href: b.href,
    icon: outletIcon(b),
    source: bylineSource(b),
  })),
];

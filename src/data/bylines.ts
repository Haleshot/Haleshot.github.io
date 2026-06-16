// Writing published elsewhere (guest posts, company blog, community blog). Newest first.
// The homepage shows the first few; the rest live here as the full record.
export interface Byline {
  title: string;
  outlet: string; // e.g. "Deepnote blog"
  date: string; // human-readable, e.g. "2026"
  href: string;
}

export const BYLINES: Byline[] = [
  {
    title: "Community write-ups on the SciPy India blog",
    outlet: "SciPy India",
    date: "2025–26",
    href: "https://scipy.in/blog/author/srihari-thyagarajan",
  },
  {
    title: "Jupyter AI: how to use AI tools in your notebooks",
    outlet: "Deepnote blog",
    date: "2026",
    href: "https://deepnote.com/blog/jupyter-ai-guide",
  },
  {
    title: "Top AI data visualization tools in 2026",
    outlet: "Deepnote blog",
    date: "2026",
    href: "https://deepnote.com/blog/ai-data-visualization-tools",
  },
  {
    title: "Tracking the ripple effect: how adverse drug events move pharma markets",
    outlet: "Deepnote blog",
    date: "2026",
    href: "https://deepnote.com/blog/tracking-the-ripple-effect-how-adverse-drug-events-move-pharma-markets",
  },
  {
    title: "Deepnote alternatives series",
    outlet: "Deepnote",
    date: "2026",
    href: "https://deepnote.com/alternatives/sagemaker",
  },
];

// Full Deepnote "alternatives" series I wrote, kept here for the record.
// Surfaced on the homepage via the grouped byline above; marimo is coming soon.
export const DEEPNOTE_ALTERNATIVES = [
  "https://deepnote.com/alternatives/sagemaker",
  "https://deepnote.com/alternatives/jupyter",
  "https://deepnote.com/alternatives/colab",
  "https://deepnote.com/alternatives/hex",
  "https://deepnote.com/alternatives/databricks",
  "https://deepnote.com/alternatives/kaggle",
  "https://deepnote.com/alternatives/mode",
  "https://deepnote.com/alternatives/vscode",
  "https://deepnote.com/alternatives/julius",
  // "https://deepnote.com/alternatives/marimo", // coming soon
] as const;

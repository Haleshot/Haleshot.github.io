// Writing published elsewhere (guest posts, company blog, community blog).
// The homepage shows the first five, so keep the ones you want there at the top.
// The posts page sorts by pubDate and slots these into the timeline; entries
// without a pubDate (ongoing series) get listed at the bottom of that page.
export interface Byline {
  title: string;
  outlet: string; // e.g. "Deepnote blog"
  date: string; // human-readable, e.g. "2026"
  href: string;
  pubDate?: string; // ISO date, e.g. "2026-10-09"
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
    pubDate: "2026-04-29",
  },
  {
    title: "Top AI data visualization tools in 2026",
    outlet: "Deepnote blog",
    date: "2026",
    href: "https://deepnote.com/blog/ai-data-visualization-tools",
    pubDate: "2026-04-16",
  },
  {
    title: "Tracking the ripple effect: how adverse drug events move pharma markets",
    outlet: "Deepnote blog",
    date: "2026",
    href: "https://deepnote.com/blog/tracking-the-ripple-effect-how-adverse-drug-events-move-pharma-markets",
    pubDate: "2026-03-11",
  },
  {
    title: "Deepnote alternatives series",
    outlet: "Deepnote",
    date: "2026",
    href: "https://deepnote.com/alternatives/sagemaker",
  },
  {
    title: "Pie charts explained: when to use one, examples, and how to make one",
    outlet: "Deepnote blog",
    date: "2026",
    href: "https://deepnote.com/blog/pie-chart",
    pubDate: "2026-11-02",
  },
  {
    title: "31 types of graphs and charts (and when to use each)",
    outlet: "Deepnote blog",
    date: "2026",
    href: "https://deepnote.com/blog/types-of-graphs",
    pubDate: "2026-10-09",
  },
  {
    title: "The 15 best data visualization tools in 2026 (free and paid)",
    outlet: "Deepnote blog",
    date: "2026",
    href: "https://deepnote.com/blog/data-visualization-tools",
    pubDate: "2026-10-08",
  },
  {
    title: "Box plots explained: when to use one, examples, and how to make one",
    outlet: "Deepnote blog",
    date: "2026",
    href: "https://deepnote.com/blog/box-plot",
    pubDate: "2026-10-08",
  },
  {
    title: "8 misleading graph examples: how to spot bad charts and fix them",
    outlet: "Deepnote blog",
    date: "2026",
    href: "https://deepnote.com/blog/misleading-graphs",
    pubDate: "2026-10-02",
  },
  {
    title: "Dot plots explained: when to use one, examples, and how to make one",
    outlet: "Deepnote blog",
    date: "2026",
    href: "https://deepnote.com/blog/dot-plot",
    pubDate: "2026-10-02",
  },
  {
    title: "IndiaFOSS 2026: the docs devroom",
    outlet: "Write the Docs India blog",
    date: "2026",
    href: "https://write-the-docs-india.github.io/blog/2026/indiafoss-2026-devroom/",
    pubDate: "2026-09-28",
  },
  {
    title: "JupyterLab vs Deepnote: a side-by-side comparison for 2026",
    outlet: "Deepnote blog",
    date: "2026",
    href: "https://deepnote.com/blog/jupyterlab-vs-deepnote",
    pubDate: "2026-08-31",
  },
  {
    title: "Hex vs Deepnote: a side-by-side comparison",
    outlet: "Deepnote blog",
    date: "2026",
    href: "https://deepnote.com/blog/hex-vs-deepnote",
    pubDate: "2026-08-31",
  },
  {
    title: "JupyterHub vs JupyterLab: they solve different problems",
    outlet: "Deepnote blog",
    date: "2026",
    href: "https://deepnote.com/blog/jupyterhub-vs-jupyterlab",
    pubDate: "2026-03-02",
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

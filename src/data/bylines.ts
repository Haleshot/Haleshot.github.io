// Writing published elsewhere (guest posts, company blog, community blog).
// The homepage groups these by outlet and shows the newest from each.
// The posts page sorts by pubDate and slots these into the timeline; entries
// without a pubDate (ongoing series) get listed at the bottom of that page.
// Set `related` to one of my post paths to show the piece under that post instead.
export interface Byline {
  title: string;
  outlet: string; // e.g. "Deepnote blog"
  date: string; // human-readable, e.g. "2026"
  href: string;
  pubDate?: string; // ISO date, e.g. "2026-10-09"
  description?: string; // one line, shown on the posts page
  related?: string; // e.g. "/posts/2026/indiafoss-2026-experience"
}

// "Deepnote blog" -> "deepnote", used as the posts page filter key
export const bylineSource = (b: Byline) =>
  b.outlet
    .replace(/ blog$/, "")
    .toLowerCase()
    .replace(/\s+/g, "-");

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
    description:
      "Trying the chat UI and the %%ai magic in a real notebook, to see where Jupyter AI helps and where it stalls.",
  },
  {
    title: "Top AI data visualization tools in 2026",
    outlet: "Deepnote blog",
    date: "2026",
    href: "https://deepnote.com/blog/ai-data-visualization-tools",
    pubDate: "2026-04-16",
    description:
      "The same small churn benchmark run across AI data visualization tools, to see how hard each one makes it to ship a wrong answer.",
  },
  {
    title: "Tracking the ripple effect: how adverse drug events move pharma markets",
    outlet: "Deepnote blog",
    date: "2026",
    href: "https://deepnote.com/blog/tracking-the-ripple-effect-how-adverse-drug-events-move-pharma-markets",
    pubDate: "2026-03-11",
    description:
      "An event study mapping adverse drug event signals to pharma stock moves, using price data and FAERS counts.",
  },
  {
    title: "Deepnote alternatives series, 12 pages so far",
    outlet: "Deepnote",
    date: "2026",
    href: "https://deepnote.com/alternatives/sagemaker",
    description:
      "Deepnote next to the tools people come from: Jupyter, JupyterLab, Colab, Hex, Databricks, Kaggle, Mode, SageMaker, VS Code, Streamlit, Julius and marimo.",
  },
  {
    title: "31 types of graphs and charts (and when to use each)",
    outlet: "Deepnote blog",
    date: "2026",
    href: "https://deepnote.com/blog/types-of-graphs",
    pubDate: "2026-10-09",
    description:
      "An interactive example for each one, and how to pick the right chart for your data.",
  },
  {
    title: "The 15 best data visualization tools in 2026 (free and paid)",
    outlet: "Deepnote blog",
    date: "2026",
    href: "https://deepnote.com/blog/data-visualization-tools",
    pubDate: "2026-10-08",
    description: "Fifteen tools compared on workflow, collaboration, AI support and pricing.",
  },
  {
    title: "Histogram vs bar graph: differences, examples, and when to use each",
    outlet: "Deepnote blog",
    date: "2026",
    href: "https://deepnote.com/blog/histogram-vs-bar-graph",
    pubDate: "2026-10-08",
    description:
      "The same dataset as a histogram and as a bar graph, and how grouping changes the question each one answers.",
  },
  {
    title: "How to make a pie chart (in Excel and other tools)",
    outlet: "Deepnote blog",
    date: "2026",
    href: "https://deepnote.com/blog/how-to-make-a-pie-chart",
    pubDate: "2026-10-08",
    description:
      "Pie charts in Excel, Google Sheets, Python, PowerPoint and Deepnote, plus labeling percentages and picking a variant.",
  },
  {
    title: "Box plots explained: when to use one, examples, and how to make one",
    outlet: "Deepnote blog",
    date: "2026",
    href: "https://deepnote.com/blog/box-plot",
    pubDate: "2026-10-08",
    description:
      "How to read a box plot's median, quartiles, and outliers, and compare distributions with worked examples.",
  },
  {
    title: "8 misleading graph examples: how to spot bad charts and fix them",
    outlet: "Deepnote blog",
    date: "2026",
    href: "https://deepnote.com/blog/misleading-graphs",
    pubDate: "2026-10-02",
    description: "Eight charts that mislead, and how to fix each one in Deepnote.",
  },
  {
    title: "Dot plots explained: when to use one, examples, and how to make one",
    outlet: "Deepnote blog",
    date: "2026",
    href: "https://deepnote.com/blog/dot-plot",
    pubDate: "2026-10-02",
    description:
      "Traditional and Cleveland dot plots, and when to reach for one over another chart.",
  },
  {
    title: "IndiaFOSS 2026 | Community booth",
    outlet: "SciPy India",
    date: "2026",
    href: "https://scipy.in/blog/2026-09-26-indiafoss-community-booth",
    pubDate: "2026-09-26",
    description:
      "Two days behind the SciPy India booth, talking to anyone who stopped by about what we're building towards in December.",
    related: "/posts/2026/indiafoss-2026-experience",
  },
  {
    title: "IndiaFOSS 2026: the docs devroom",
    outlet: "Write the Docs India blog",
    date: "2026",
    href: "https://write-the-docs-india.github.io/blog/2026/indiafoss-2026-devroom/",
    pubDate: "2026-09-28",
    description: "Ten talks in one morning, and the chapter's restart.",
    related: "/posts/2026/indiafoss-2026-experience",
  },
  {
    title: "JupyterLab vs Deepnote: a side-by-side comparison for 2026",
    outlet: "Deepnote blog",
    date: "2026",
    href: "https://deepnote.com/blog/jupyterlab-vs-deepnote",
    pubDate: "2026-08-31",
    description: "Where the two notebooks overlap and where they differ, side by side.",
  },
  {
    title: "Hex vs Deepnote: a side-by-side comparison",
    outlet: "Deepnote blog",
    date: "2026",
    href: "https://deepnote.com/blog/hex-vs-deepnote",
    pubDate: "2026-08-31",
    description:
      "Hex and Deepnote both combine code, data, AI, and outputs in a browser-based workspace, but they're built for different kinds of work.",
  },
  {
    title: "JupyterHub vs JupyterLab: they solve different problems",
    outlet: "Deepnote blog",
    date: "2026",
    href: "https://deepnote.com/blog/jupyterhub-vs-jupyterlab",
    pubDate: "2026-03-02",
    description:
      "JupyterLab is the interface you work in. JupyterHub manages access for multiple users.",
  },
];

// Full Deepnote "alternatives" series I wrote, kept here for the record.
// Surfaced on the homepage via the grouped byline above.
export const DEEPNOTE_ALTERNATIVES = [
  "https://deepnote.com/alternatives/sagemaker",
  "https://deepnote.com/alternatives/jupyter",
  "https://deepnote.com/alternatives/colab",
  "https://deepnote.com/alternatives/hex",
  "https://deepnote.com/alternatives/databricks",
  "https://deepnote.com/alternatives/kaggle",
  "https://deepnote.com/alternatives/mode",
  "https://deepnote.com/alternatives/vscode",
  "https://deepnote.com/alternatives/juliusai",
  "https://deepnote.com/alternatives/jupyter-lab",
  "https://deepnote.com/alternatives/streamlit",
  "https://deepnote.com/compare/alternatives/marimo",
] as const;

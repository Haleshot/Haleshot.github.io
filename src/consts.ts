// Place any global data in this file.
// You can import this data from anywhere in your site by using the `import` keyword.

interface SocialLink {
  href: string;
  label: string;
}

interface Site {
  website: string;
  author: string;
  profile: string;
  desc: string;
  title: string;
  ogImage: string;
  lightAndDarkMode: boolean;
  postPerIndex: number;
  postPerPage: number;
  scheduledPostMargin: number;
  showArchives: boolean;
  showBackButton: boolean;
  editPost: {
    enabled: boolean;
    text: string;
    url: string;
  };
  dynamicOgImage: boolean;
  lang: string;
  timezone: string;
}

// Site configuration
export const SITE: Site = {
  website: "https://haleshot.github.io/",
  author: "Srihari Thyagarajan",
  profile: "https://haleshot.github.io/about",
  desc: "Technical Writer at Deepnote. Writing about developer tools, documentation, and technical content — especially notebooks and open-source communities.",
  title: "Haleshot",
  ogImage: "srihari-avatar.png",
  lightAndDarkMode: true,
  postPerIndex: 10,
  postPerPage: 10,
  scheduledPostMargin: 15 * 60 * 1000,
  showArchives: false,
  showBackButton: true,
  editPost: {
    enabled: true,
    text: "Edit on GitHub",
    url: "https://github.com/Haleshot/Haleshot.github.io/edit/main/",
  },
  dynamicOgImage: true,
  lang: "en",
  timezone: "Asia/Kolkata",
};

export const SITE_TITLE = SITE.title;
export const SITE_DESCRIPTION = SITE.desc;

// Navigation links
export const NAV_LINKS: SocialLink[] = [
  {
    href: "/",
    label: "Blog",
  },
  {
    href: "/about",
    label: "About",
  },
  {
    href: "/now",
    label: "Now",
  },
  {
    href: "/ai",
    label: "AI",
  },
];

// Social media links
export const SOCIAL_LINKS: SocialLink[] = [
  {
    href: "https://github.com/Haleshot",
    label: "GitHub",
  },
  {
    href: "https://twitter.com/hari_leo03",
    label: "Twitter",
  },
  {
    href: "https://bsky.app/profile/haleshot.bsky.social",
    label: "BlueSky",
  },
  {
    href: "/rss.xml",
    label: "RSS",
  },
];

// Icon map for social media
export const ICON_MAP: Record<string, string> = {
  GitHub: "github",
  Twitter: "twitter",
  BlueSky: "bsky",
  RSS: "rss",
  Email: "mail",
};

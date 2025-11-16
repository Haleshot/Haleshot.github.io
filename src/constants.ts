import { SITE } from "./consts";

export const SOCIALS = [
  {
    name: "Github",
    href: "https://github.com/Haleshot",
    linkTitle: ` ${SITE.title} on Github`,
    icon: "github",
    active: true,
  },
  {
    name: "X",
    href: "https://x.com/hari_leo03",
    linkTitle: `${SITE.title} on X`,
    icon: "twitter",
    active: true,
  },
  {
    name: "BlueSky",
    href: "https://bsky.app/profile/haleshot.bsky.social",
    linkTitle: `${SITE.title} on BlueSky`,
    icon: "bluesky",
    active: true,
  },
  {
    name: "Mastodon",
    href: "https://mastodon.social/@haleshot",
    linkTitle: `${SITE.title} on Mastodon`,
    icon: "mastodon",
    active: true,
  },
  {
    name: "LinkedIn",
    href: "https://www.linkedin.com/in/srihari-thyagarajan/",
    linkTitle: `${SITE.title} on LinkedIn`,
    icon: "linkedin",
    active: true,
  },
  {
    name: "Mail",
    href: "mailto:hari.leo03@gmail.com",
    linkTitle: `Send an email to ${SITE.title}`,
    icon: "mail",
    active: true,
  },
] as const;

export const SHARE_LINKS = [
  {
    name: "X",
    href: "https://x.com/intent/post?url=",
    linkTitle: `Share this post on X`,
    icon: "twitter",
  },
  {
    name: "BlueSky",
    href: "https://bsky.app/intent/compose?text=",
    linkTitle: `Share this post on BlueSky`,
    icon: "bluesky",
  },
  {
    name: "LinkedIn",
    href: "https://www.linkedin.com/sharing/share-offsite/?url=",
    linkTitle: `Share this post on LinkedIn`,
    icon: "linkedin",
  },
  {
    name: "WhatsApp",
    href: "https://wa.me/?text=",
    linkTitle: `Share this post via WhatsApp`,
    icon: "whatsapp",
  },
  {
    name: "Facebook",
    href: "https://www.facebook.com/sharer.php?u=",
    linkTitle: `Share this post on Facebook`,
    icon: "facebook",
  },
  {
    name: "Telegram",
    href: "https://t.me/share/url?url=",
    linkTitle: `Share this post via Telegram`,
    icon: "telegram",
  },
  {
    name: "Pinterest",
    href: "https://pinterest.com/pin/create/button/?url=",
    linkTitle: `Share this post on Pinterest`,
    icon: "pinterest",
  },
  {
    name: "Mail",
    href: "mailto:?subject=See%20this%20post&body=",
    linkTitle: `Share this post via email`,
    icon: "mail",
  },
] as const;

// Giscus configuration for comments
// Configuration from https://giscus.app
export const GISCUS = {
  repo: "Haleshot/haleshot.github.io",
  repoId: "R_kgDOLUQrqQ",
  category: "General",
  categoryId: "DIC_kwDOLUQrqc4CgZZJ",
  mapping: "pathname" as const,
  strict: "0" as const,
  reactionsEnabled: "1" as const,
  emitMetadata: "0" as const,
  inputPosition: "bottom" as const,
  lang: "en",
  loading: "lazy" as const,
};

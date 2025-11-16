import type { APIRoute } from "astro";

export const GET: APIRoute = async () => {
  const markdownContent = `# Srihari Thyagarajan (@Haleshot)

Technical Writer at Deepnote. Writing about developer tools, documentation, and technical content — with a soft spot for notebooks and open-source communities.

## Navigation

- [About](/about.md)
- [Recent Posts](/posts.md)
- [RSS Feed](/rss.xml)

## Links

- Twitter: [@hari_leo03](https://twitter.com/hari_leo03)
- GitHub: [@Haleshot](https://github.com/Haleshot)
- LinkedIn: [srihari-thyagarajan](https://www.linkedin.com/in/srihari-thyagarajan/)
- Mastodon: [@haleshot@mastodon.social](https://mastodon.social/@haleshot)
- BlueSky: [@haleshot.bsky.social](https://bsky.app/profile/haleshot.bsky.social)
- Email: hari.leo03@gmail.com

---

*This is the markdown-only version of haleshot.github.io. Visit [haleshot.github.io](https://haleshot.github.io) for the full experience.*`;

  return new Response(markdownContent, {
    status: 200,
    headers: {
      "Content-Type": "text/markdown; charset=utf-8",
      "Cache-Control": "public, max-age=3600",
    },
  });
};

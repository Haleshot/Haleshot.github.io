# Haleshot's Personal Website

This is the source code for my personal website, built with [Astro](https://astro.build) and deployed on GitHub Pages.

## About

I'm Srihari Thyagarajan (Haleshot), a Technical Writer at Deepnote passionate about developer tools, documentation, and open-source communities. This website hosts my blog and information about my work in technical writing, developer advocacy, and community building.

## Project Structure

```text
├── public/               # Static assets (images, fonts, favicon)
│   ├── assets/          # Images for blog posts
│   └── fonts/           # Web fonts
├── src/
│   ├── assets/          # Icons and images used in components
│   ├── components/      # Reusable UI components
│   │   └── ui/          # React components
│   ├── content/         # Content collections
│   │   └── blog/        # Blog posts in Markdown format (organized by year)
│   ├── layouts/         # Page layouts and templates
│   ├── pages/           # Routes and pages
│   ├── styles/          # Global styles and CSS
│   └── utils/           # Utility functions
├── astro.config.mjs     # Astro configuration
├── vercel.json          # Vercel deployment and CSP configuration
├── package.json         # Project dependencies and scripts
├── tailwind.config.mjs  # Tailwind CSS configuration
└── LICENSE              # Dual license (CC BY 4.0 + MIT)
```

## Commands

| Command                | Action                                      |
| :--------------------- | :------------------------------------------ |
| `npm install`          | Installs dependencies                       |
| `npm run dev`          | Starts local dev server at `localhost:4321` |
| `npm run build`        | Build the production site to `./dist/`      |
| `npm run preview`      | Preview the build locally, before deploying |

## Deployment

This site is deployed on GitHub Pages. Push to the `main` branch to trigger automatic deployment.

## License

This repository uses dual licensing:

- **Documentation & Blog Posts**: Licensed under [CC BY 4.0](http://creativecommons.org/licenses/by/4.0/)
- **Code & Code Snippets**: Licensed under the [MIT License](LICENSE)

See the [LICENSE](LICENSE) file for full details.

## Attribution & Inspiration

This website was inspired by [Peter Steinberger's personal website](https://steipete.me) ([GitHub](https://github.com/steipete/steipete.me)). I discovered Peter's site while exploring personal blogs and portfolios of developers working in the open-source space. What caught my attention was the clean, modern design, conversational writing style, and how he balanced technical content with community-focused work — something I deeply relate to in my own journey with developer relations and community building.

Peter's site itself is built with [Astro](https://astro.build) and uses the excellent [AstroPaper theme](https://astro-paper.pages.dev/) created by [Sat Naing](https://github.com/satnaing). I loved the approach so much that I decided to adapt it for my own use, customizing it to reflect my work with notebooks, technical writing, and open-source collaboration.

Special thanks to:

- **[Peter Steinberger (@steipete)](https://github.com/steipete)** for the inspiration and demonstrating how to build a developer-focused personal site with personality
- **[Sat Naing (@satnaing)](https://github.com/satnaing)** for creating and maintaining the [AstroPaper theme](https://astro-paper.pages.dev/) that powers this site

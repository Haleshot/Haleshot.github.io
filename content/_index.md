---
# Leave the homepage title empty to use the site title
title: ''
date: 2022-10-24
type: landing

sections:
  - block: about.biography
    id: about
    content:
      title: Biography
      # Choose a user profile to display (a folder name within `content/authors/`)
      username: admin
  - block: experience
    content:
      title: Experience
      # Date format for experience
      #   Refer to https://docs.hugoblox.com/customization/#date-format
      date_format: Jan 2006
      # Experiences.
      #   Add/remove as many `experience` items below as you like.
      #   Required fields are `title`, `company`, and `date_start`.
      #   Leave `date_end` empty if it's your current employer.
      #   Begin multi-line descriptions with YAML's `|2-` multi-line prefix.
      items:
        - title: Intern | AI/ML and Developer Advocacy
          company: marimo.io
          company_url: 'https://marimo.io?ref=https://haleshot.github.io/'
          company_logo: marimo
          location: Remote, SF, United States
          date_start: '2025-01-01'
          date_end: ''
          description: |2-
              Joined marimo to work on their reactive Python notebook platform; ended up getting really into the community and adoption side of things.

              **Technical Contributions & Development**:
              * Contributed [37+ merged PRs](https://github.com/marimo-team/marimo/commits?author=Haleshot) to the marimo OSS repo — enhanced [snippets system](https://github.com/marimo-team/marimo/pull/3709) with 18 practical code examples that improved developer onboarding
              * Developed [mo.image_compare() widget](https://github.com/marimo-team/marimo/pull/5091) for interactive before/after visualizations; now used across the community for data comparison workflows  
              * Identified and resolved [Windows-specific marimo islands bugs](https://github.com/marimo-team/marimo/pull/3242), improving cross-platform developer experience
              * Implemented AI integrations including [Groq chat support](https://github.com/marimo-team/marimo/pull/2757) and [DeepSeek configuration guide](https://github.com/marimo-team/marimo/pull/3597); developed [QoL package installation UX improvements](https://github.com/marimo-team/marimo/pull/3961)

              **Community & Partnership Development**:
              * Spearheaded development of 50+ interactive educational notebooks across Python fundamentals, probability theory and linear algebra — launched comprehensive [probability series](https://www.linkedin.com/posts/srihari-thyagarajan_interactive-computational-sets-activity-7315944898711453696-zYqD) based on [Stanford CS109](https://chrispiech.github.io/probabilityForComputerScientists/en/index.html) with 20+ interactive notebooks
              * Launched [Deep-ML integration](https://www.linkedin.com/posts/srihari-thyagarajan_ai-ml-ai-activity-7283552975829037057-v5MK) with interactive learning tab via [deepml-notebooks](https://github.com/marimo-team/deepml-notebooks); established [HuggingFace course integration](https://huggingface.co/learn/llm-course/en/chapter12/4?fw=pt), positioning marimo within major ML education ecosystems
              * Established multi-platform [collaboration](https://www.linkedin.com/posts/srihari-thyagarajan_momentum-collaboration-interactive-activity-7317451893771751424-iCfy) with [GroundZeroAI](https://x.com/groundzero_ai/status/1909145716280176813) for Linear Algebra course; achieved 180+ GitHub stars; launched [landing page](https://marimo-team.github.io/learn/) for quick WASM notebook previews
              * Added marimo to [CNCF landscape](https://github.com/cncf/landscape/pull/4208), increasing visibility in cloud-native developer communities

              **Developer Relations & Outreach**:
              * Led community engagement via [spotlight showcases](https://github.com/marimo-team/spotlights); delivered [UCSD workshop](https://www.linkedin.com/posts/srihari-thyagarajan_nba-wasm-data-activity-7304408832158400512-HQuG?utm_source=share&utm_medium=member_desktop&rcm=ACoAADSJzvgBkjBd85IWDyUWA6ttzq8B-NDq-Hs) on NBA data analysis, pursued enterprise adoption outreach — expanded platform reach across technical communities
              * Spearheaded technical outreach initiatives to ML practitioners and teams — initiated conversations with engineers from various companies for Streamlit/Jupyter migration to marimo
              * Established multi-platform [partnership with GroundZeroAI](https://x.com/groundzero_ai/status/1909145716280176813) for Linear Algebra course development

              Expanded platform adoption across educational institutions, enterprise teams & open-source communities

              Skills: Python, Developer Relations, Technical Writing, Platform Development, Community Building, Open Source Development, AI/ML Integration, Partnership Development

        - title: Ambassador
          company: marimo.io
          company_url: 'https://marimo.io?ref=https://haleshot.github.io/'
          company_logo: marimo
          location: Remote, United States
          date_start: '2024-09-01'
          date_end: ''
          description: |2-
              **Community Building & Content Creation**:
              * Develop educational content and tutorials that help developers transition from traditional notebooks to reactive programming paradigms
              * Manage and contribute to the [marimo learn repository](https://github.com/marimo-team/learn) and [deepml-notebooks](https://github.com/Haleshot/deepml-notebooks), creating interactive learning resources that make complex concepts accessible
              * Showcase community projects through weekly [spotlight features](https://github.com/marimo-team/spotlights), highlighting innovative applications and encouraging diverse use cases

              **Developer Engagement**:
              * Create practical examples and relevant docs that addresses real developer pain points in data science workflows
              * Support new contributors through code reviews, mentorship & guidance on effective notebook design patterns

              **Ecosystem Growth**:
              * Collaborate with educational platforms and open-source projects to expand marimo's integration within the broader developer tools ecosystem
              * Advocate for reactive notebook adoption through technical demonstrations and community presentations

              Skills: Python, Developer Relations, Technical Writing, Community Building, Open Source Development, Educational Content Creation

        - title: Research Intern
          company: SVKM's NMIMS Mukesh Patel School of Technology Management & Engineering
          company_url: 'https://engineering.nmims.edu/'
          company_logo: nmims
          location: Mumbai, Maharashtra, India
          date_start: '2024-07-01'
          date_end: '2024-11-17'
          description: |2-
              **Capstone Project:** [MathMate | Multimodal AI Assistant for Math Learning](https://github.com/Haleshot/Capstone_Project/)
              <br>
              **Mentor:** Dr. Vaishali Kulkarni

              Developing an innovative LLM-based project specialized in mathematical reasoning and problem-solving.

              Key accomplishments:
              * Implemented model merging techniques using [mergekit](https://github.com/cg123/mergekit), combining NuminaMath-7B-TIR and DeepSeek-Prover-V1.5-RL models.
              * Created [Mathmate-7B-DELLA](https://huggingface.co/Haleshot/Mathmate-7B-DELLA), a 6.91B parameter model optimized for mathematical tasks.
              * Conducted model evaluation using LLM AutoEval with the Nous dataset.
              * Applied ORPO fine-tuning on a specialized math dataset to enhance model performance.
              * Utilized cutting-edge tools including [Hugging Face](https://huggingface.co/), [Lightning.ai](https://lightning.ai/), and [Weights & Biases](https://wandb.ai/) for model development and analysis.

              This project aims to advance AI capabilities in mathematical reasoning, potentially revolutionizing how students and researchers approach complex mathematical problems.

              Skills: Python, Research, Large Language Models (LLM), AI, Machine Learning, Computer Vision, GUI, Product Ideation and Development, Cloud Applications, Model Deployment, Git, GitHub

        - title: Research Intern
          company: Polymath Jr. program
          company_url: 'https://geometrynyc.wixsite.com/polymathreu'
          company_logo: polymath
          location: Remote, United States, India.
          date_start: '2024-06-17'
          date_end: '2024-08-15'
          description: |2-
              * **[Notabot Studio Project](https://notabot.ai/)**
                - **Role:** Project organizer, Game Developer
                - **Key Responsibilities**:
                    - **Technologies Used:** PlayCanvas
                    - **Ideation:** Contributed ideas; one selected for development.
                    - **Team Leadership:** Leading a team; coordinated project efforts.
                    - **Game Development:** Designing mechanics to teach math concepts; using PlayCanvas.
                - **Achievements**:
                    - Developing a project combining education and entertainment.
          
        - title: AI Internship
          company: Digital India Corporation (DIC)
          company_url: 'https://dic.gov.in/'
          company_logo: dic
          location: Remote, New Delhi, India.
          date_start: '2024-05-01'
          date_end: '2024-07-01'
          description: |2-
              * Roles and Responsibilities:
                * Analyzed product listing workflow and proposed AI solutions for [IndiaHandMade E-commerce Platform](https://www.indiahandmade.com/).
                * Integrated image captioning and language models for automated product descriptions.
                * Implemented audio transcription model to enable multilingual product descriptions.
                * Developed APIs using FastAPI for seamless integration with the existing platform.
                * Collaborated with the team for deployment on Ubuntu server, resolving dependencies.
                * Automated product listing process, enhancing vendor experience.
                * Enabled multilingual product descriptions through audio transcription.
                * Streamlined API development and deployment using open-source technologies.
                * Demonstrated problem-solving, optimization, and open-source advocacy skills.
                * Python, FastAPI, HuggingFace Models (`Salesforce/blip-image-captioning-large`, Ollama/Groq Llama3, Whisper), API Development, Ubuntu Server Deployment.

        - title: Customer Service, Technical Support, QA (Freelance)
          company: Gif Your Game
          company_url: 'https://gifyourgame.com/'
          company_logo: gyg
          location: Remote, Santa Monica, California
          date_start: '2020-09-01'
          date_end: '2024-04-01'
          description: |2-
              * Quality Assurance (QA) and Testing:
                * Maintaining user logs about issues/feedback.
                * Ensure products meet customer expectations and demand.
                * Create reports documenting errors and issues for fixing.
                * Perform Quality testing for the app and documenting respective errors/bugs for the development team.
              * Customer Service, Technical Support:
                * Identify and address customer needs with a goal of complete satisfaction.
                * Follow company communications guidelines and procedures under minimal supervision.
                * Research information using available resources to satisfy customer inquiries.
                * Identify and address customer needs with a goal of complete satisfaction.
                * Follow company communications guidelines and procedures under minimal supervision.
                * Research information using available resources to satisfy customer inquiries.
                * Build rapport with customers by engaging with them in an inviting, friendly, and professional manner, 
                  to deliver exceptional experiences nurture lasting relationships.
                * Assist with moderation of content the Discord Server (with over 120,000+ members).
        - title: Intern 
          company: Engagely.ai
          company_url: 'https://engagely.ai'
          company_logo: engagely
          location: Remote, Mumbai.
          date_start: '2023-05-01'
          date_end: '2023-06-01'
          description:  |2-
              * Hands-on Testing Exposure:
                * Contributed to RCM testing.
                * Learned bug detection, QA collaboration, and documentation.
              * Practical Codebase Familiarity:
                * Access to GitLab repository.
                * Explored Python-Flask, Docker, docker-compose.
              * Collaborative Coding Insights:
                * Introduction to Git participation.
                * Grasped pull requests, issue management.
                * Engagely.ai internship provided real-world testing insights, coding exposure, and collaborative skills enhancement.          
    design:
      columns: '2'
  - block: portfolio
    id: posts
    content:
      title: Blog Posts
      subtitle: 'Technical Content & Learning Journey'
      text: |-
        {{% callout note %}}
        **Content Strategy**: Currently focused on creating high-quality technical content for developer advocacy and community building. New in-depth articles and tutorials are coming soon!
        {{% /callout %}}
      filters:
        folders:
          - post
      # Default filter index (e.g. 0 corresponds to the first `filter_button` instance below).
      default_button_index: 0
      # Filter toolbar (optional).
      # Add or remove as many filters (`filter_button` instances) as you like.
      # To show all items, set `tag` to "*".
      # To filter by a specific tag, set `tag` to an existing tag name.
      # To remove the toolbar, delete the entire `filter_button` block.
      buttons:
        - name: Featured Posts
          tag: 'Featured Blogs'
        - name: Archived Posts
          tag: '*'
    design:
      # Choose how many columns the section has. Valid values: '1' or '2'.
      columns: '2'
      view: card
      # For Showcase view, flip alternate rows?
      flip_alt_rows: false
  - block: portfolio
    id: projects
    content:
      title: Projects
      filters:
        folders:
          - project
      # Default filter index (e.g. 0 corresponds to the first `filter_button` instance below).
      default_button_index: 0
      # Filter toolbar (optional).
      # Add or remove as many filters (`filter_button` instances) as you like.
      # To show all items, set `tag` to "*".
      # To filter by a specific tag, set `tag` to an existing tag name.
      # To remove the toolbar, delete the entire `filter_button` block.
      buttons:
        - name: Machine Learning
          tag: Machine Learning
        - name: Deep Learning
          tag: Deep Learning
        - name: Image Processing
          tag: Image Processing
        - name: Open Source
          tag: Open Source
        - name: Miscellaneous
          tag: Misc
    design:
      # Choose how many columns the section has. Valid values: '1' or '2'.
      columns: '1'
      view: showcase
      # For Showcase view, flip alternate rows?
      flip_alt_rows: false
  - block: collection
    id: talks
    content:
      title: Recent & Upcoming Talks
      filters:
        folders:
          - event
    design:
      columns: '2'
      view: compact
  - block: tag_cloud
    content:
      title: Popular Topics
    design:
      columns: '2'
  - block: contact
    id: contact
    content:
      title: Contact
      subtitle:
      text: |-
        Send an email!
      appointment_url: 'https://calendly.com/hari-leo03'
      contact_links:
        - icon: twitter
          icon_pack: fab
          name: DM Me
          link: 'https://x.com/hari_leo03'
      # Automatically link email and phone or display as text?
      autolink: true
      # Email form provider
      form:
        provider: formspree
        formspree:
          id: xblrlwop
        netlify:
          # Enable CAPTCHA challenge to reduce spam?
          captcha: true
    design:
      columns: '2'
---

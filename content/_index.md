---
# Leave the homepage title empty to use the site title
title: ''
date: 2022-10-24
type: landing

sections:
  - block: hero
    demo: true # Only display this section in the Hugo Blox Builder demo site
    content:
      title: Hugo Academic Theme
      image:
        filename: hero-academic.png
      cta:
        label: '**Get Started**'
        url: https://hugoblox.com/templates/
      cta_alt:
        label: Ask a question
        url: https://discord.gg/z8wNYzb
      cta_note:
        label: >-
          <div style="text-shadow: none;"><a class="github-button" href="https://github.com/HugoBlox/hugo-blox-builder" data-icon="octicon-star" data-size="large" data-show-count="true" aria-label="Star">Star Hugo Blox Builder</a></div><div style="text-shadow: none;"><a class="github-button" href="https://github.com/HugoBlox/theme-academic-cv" data-icon="octicon-star" data-size="large" data-show-count="true" aria-label="Star">Star the Academic template</a></div>
      text: |-
        **Generated by Hugo Blox Builder - the FREE, Hugo-based open source website builder trusted by 500,000+ sites.**

        **Easily build anything with blocks - no-code required!**

        From landing pages, second brains, and courses to academic resumés, conferences, and tech blogs.

        <!--Custom spacing-->
        <div class="mb-3"></div>
        <!--GitHub Button JS-->
        <script async defer src="https://buttons.github.io/buttons.js"></script>
    design:
      background:
        gradient_end: '#1976d2'
        gradient_start: '#004ba0'
        text_color_light: true
  - block: about.biography
    id: about
    content:
      title: Biography
      # Choose a user profile to display (a folder name within `content/authors/`)
      username: admin
  - block: skills
    content:
      title: Skills
      text: ''
      # Choose a user to display skills from (a folder name within `content/authors/`)
      username: admin
    design:
      columns: '1'
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
        - title: Ambassador
          company: marimo
          company_url: 'https://marimo.io?ref=https://haleshot.github.io/'
          company_logo: marimo
          location: Remote, United States
          date_start: '2024-09-01'
          date_end: ''
          description: |2-
              As a [marimo Ambassador](https://marimo.io/ambassadors?ref=https://haleshot.github.io/), I contribute to the growth and engagement of the AI/ML and developer relations community through:
              <br>
              * **Content Creation:** 
                - Regularly share tutorials, examples, and tips on marimo's tools and features.
                - Create resources for developers and data scientists using AI/ML notebooks with marimo.
              <br>
              * **Community Contributions:** 
                - Manage and contribute to the [marimo Spotlights GitHub repository](https://github.com/marimo-team/spotlights).
                - Showcase weekly community projects demonstrating creative uses of marimo notebooks.
                - Encourage diverse applications and community contributions.
              <br>
              * **Event Participation:** 
                - Assist in organizing and participating in community events, including quarterly calls and weekly spotlights.
              <br>
              * **Educational Content:** 
                - Develop tutorials, examples, and best practices for marimo usage.
                - Offer tips to enhance productivity in AI/ML workflows.
                - Core maintainer of [marimo-tutorials GitHub repository](https://github.com/Haleshot/marimo-tutorials) where marimo notebook implementations spanning various domains are stored (to serve as a good reference point for newcomers and experienced users alike).
              

              Skills: Python, Developer Relations, Communication, Community Outreach and Engagement, Content Creation, Open-Source Development, AI/ML, Data Science, Notebooks

        - title: Research Intern
          company: SVKM's NMIMS Mukesh Patel School of Technology Management & Engineering
          company_url: 'https://engineering.nmims.edu/'
          company_logo: nmims
          location: Mumbai, Maharashtra, India
          date_start: '2024-07-01'
          date_end: ''
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
          date_end: ''
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
  - block: accomplishments
    content:
      # Note: `&shy;` is used to add a 'soft' hyphen in a long heading.
      title: 'Accomplish&shy;ments'
      subtitle:
      # Date format: https://docs.hugoblox.com/customization/#date-format
      date_format: Jan 2006
      # Accomplishments.
      #   Add/remove as many `item` blocks below as you like.
      #   `title`, `organization`, and `date_start` are the required parameters.
      #   Leave other parameters empty if not required.
      #   Begin multi-line descriptions with YAML's `|2-` multi-line prefix.
      items:
        - certificate_url: https://drive.google.com/file/d/1rbOJu0YSHAM7zEe2Dnp95vMAslhRY1qa/view?usp=sharing
          date_end: ''
          date_start: '2023-07-14'
          description: 'Introduction to Java Programming'
          icon: nmims
          icon_pack: custom
          organization: NMIMS
          organization_url: https://www.nmims.edu
          title: Java Programming
          url: ''
        # - certificate_url: https://www.edx.org
        #   date_end: ''
        #   date_start: '2021-01-01'
        #   description: Formulated informed blockchain models, hypotheses, and use cases.
        #   icon: edx
        #   organization: edX
        #   organization_url: https://www.edx.org
        #   title: Blockchain Fundamentals
        #   url: https://www.edx.org/professional-certificate/uc-berkeleyx-blockchain-fundamentals
        # - certificate_url: https://www.datacamp.com
        #   date_end: '2020-12-21'
        #   date_start: '2020-07-01'
        #   description: ''
        #   icon: datacamp
        #   organization: DataCamp
        #   organization_url: https://www.datacamp.com
        #   title: 'Object-Oriented Programming in R'
        #   url: ''
    design:
      columns: '2'
  - block: collection
    id: posts
    content:
      title: Recent Posts
      subtitle: ''
      text: ''
      # Choose how many pages you would like to display (0 = all pages)
      count: 5
      # Filter on criteria
      filters:
        folders:
          - post
        author: ""
        category: ""
        tag: ""
        exclude_featured: false
        exclude_future: false
        exclude_past: false
        publication_type: ""
      # Choose how many pages you would like to offset by
      offset: 0
      # Page order: descending (desc) or ascending (asc) date.
      order: desc
    design:
      # Choose a layout view
      view: compact
      columns: '2'
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
        - name: All
          tag: '*'
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
  - block: markdown
    content:
      title: Gallery
      subtitle: ''
      text: |-
        {{< gallery album="demo" >}}
    design:
      columns: '1'
  - block: collection
    id: featured
    content:
      title: Featured Publications
      filters:
        folders:
          - publication
        featured_only: true
    design:
      columns: '2'
      view: card
  - block: collection
    content:
      title: Recent Publications
      text: |-
        {{% callout note %}}
        Quickly discover relevant content by [filtering publications](./publication/).
        {{% /callout %}}
      filters:
        folders:
          - publication
        exclude_featured: true
    design:
      columns: '2'
      view: citation
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

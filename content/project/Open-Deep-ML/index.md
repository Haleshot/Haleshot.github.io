---
title: Open-Deep-ML | A CP platform offering algorithmic problems for ML practitioners
summary: Significant contributions to the [DML-OpenProblem repository](https://github.com/Open-Deep-ML/DML-OpenProblem), which powers the [deep-ml.com](https://www.deep-ml.com/) website. These contributions include bug fixes, documentation improvements, and future plans for enhancing the platform's functionality and user experience.
tags:
 - Deep Learning
 - Machine Learning
 - Open Source
 - Education
 - Problem Solving
date: '2024-10-20T00:00:00Z'

image:
 caption: Deep-ML
 focal_point: Smart

links:
 - icon: github
   icon_pack: fab
   name: GitHub
   url: 'https://github.com/Open-Deep-ML/DML-OpenProblem'
 - icon: github
   icon_pack: fab
   name: Personal Deep-ML Repo
   url: 'https://github.com/Haleshot/Deep-ML'

slides: ''

external_link: ''

---

As a B.Tech AI senior undergrad, I'm always on the lookout for resources that offer AI/ML/DL problems from scratch. Enter [deep-ml.com](https://github.com/Open-Deep-ML/DML-OpenProblem) - my latest obsession, courtesy of the TL;DR Newsletter (shoutout to my favorite "procrastination enabler"!). It's like finding a goldmine for an AI nerd like me, perfectly aligning with my *totally healthy* habit of curating resources for future projects and contributions.

{{% callout note %}}
The DML-OpenProblem repository is an open-source collection of problems focused on linear algebra, machine learning, and deep learning. It powers the [deep-ml.com](https://github.com/Open-Deep-ML/DML-OpenProblem), providing a platform for solving problems from scratch and offering a robust learning experience.
{{% /callout %}}

My Contributions:

1. Fixed bold text highlighting in the Linear Regression problem (Gradient Descent) section.
   PR: [#40](https://github.com/Open-Deep-ML/DML-OpenProblem/pull/40)

2. Added line breaks and improved HTML syntax in the Learn section.
   PR: [#45](https://github.com/Open-Deep-ML/DML-OpenProblem/pull/45)

3. Fixed Matrix transformation problem description rendering and added a test case.
   PR: [#53](https://github.com/Open-Deep-ML/DML-OpenProblem/pull/53)

4. Improved K-means clustering problem (Q_17) with better HTML syntax, clearer description, and additional test cases.
   PR: [#58](https://github.com/Open-Deep-ML/DML-OpenProblem/pull/58)

Future Plans:
- Set up CI/CD for automating website updates from the repository.
- Contribute activation functions from [ML-From-Scratch repository](https://github.com/eriklindernoren/ML-From-Scratch/blob/master/mlfromscratch/deep_learning/activation_functions.py).
- Add [video explanations](https://x.com/real_deep_ml/status/1846537201846214746) for solved problems.
- Integrate marimo with deep-ml.com (stay tuned!).

The marimo integration is particularly exciting. As an active user and [ambassador for marimo](https://marimo.io/ambassadors), I see great potential in linking these platforms. The notebook-based implementation could expand the types of problems on deep-ml.com, making it more industry-oriented. I've initiated discussions with both the marimo team and deep-ml contributors about this integration.

```mermaid
sequenceDiagram
    participant H as Haleshot
    participant DML as Deep-ML Maintainer
    participant M as Marimo Team
    H->>DML: Suggest marimo integration
    DML->>H: Request more information
    H->>DML: Provide references and resources
    DML->>H: Express interest in integration
    H->>M: Introduce Deep-ML project
    M->>DML: Offer collaboration and support
    Note over H,M: Ongoing discussions<br/>for integration
```

This project not only allows me to contribute to an educational platform but also bridges my interests in deep learning and open-source collaboration. It's a perfect blend of problem-solving, community engagement, and technological integration.

# Next Steps:
 - [ ] Continue contributing problem implementations and explanations.
 - [ ] Assist in setting up .github folder with PR templates and workflows.
 - [ ] Collaborate on marimo integration with both teams.

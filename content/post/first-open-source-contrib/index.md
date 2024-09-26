---
title: "My Journey into Open Source: Solving an ArviZ Plotting Issue in Marimo"
subtitle: "From Curating Resources to Making Meaningful Contributions"
summary: "Explore my path from a curious developer to an open-source contributor, culminating in fixing a significant plotting issue for the Marimo project."
date: '2024-09-26T00:00:00Z'
lastmod: '2024-09-26T00:00:00Z'
draft: false
featured: true
commentable: true
image:
  focal_point: 'Center'
  placement: 2
  preview_only: false
authors:
  - admin
tags:
  - Open Source
  - Python
  - Data Visualization
  - Marimo
  - ArviZ
categories:
  - Technology
  - Programming
  - Data Science
  - Open Source
projects:
  - open-source-contributions
---

{{< toc >}}

## Exciting Update: Becoming a Marimo Ambassador!

Before diving into my open-source journey, I'm thrilled to share some exciting news: I've recently become an ambassador for [Marimo](https://marimo.io)! 

{{% callout note %}}
As a Marimo Ambassador, I contribute to the growth of the AI/ML and developer relations community through content creation, community engagement, and event participation.
{{% /callout %}}

I'm particularly excited about managing the [Marimo Spotlights GitHub repository](https://github.com/marimo-team/spotlights), where we showcase weekly community projects that demonstrate creative uses of Marimo notebooks.

For more details about my role and Marimo's ambassador program, check out my [LinkedIn post](https://www.linkedin.com/posts/srihari-thyagarajan_marimo-framework-ambassador-activity-7242540448949428225-wSqS?utm_source=share&utm_medium=member_desktop) and the [Marimo Ambassadors webpage](https://marimo.io/ambassadors).

## My Open Source Journey

Open source has been a passion of mine ever since I was introduced to Git and GitHub. My journey has evolved through several phases:

1. **The Beginner Phase**: Publishing random programs and projects with minimal documentation and structure.
2. **Learning Good Practices**: Improving existing projects with better documentation, feature enhancements, and proper directory structures.
3. **Active Curation**: Regularly going through newsletters like TLDR and Rundown AI to collect and organize resources for personal projects and potential contributions.

> "I feel that's what I'm good at - curating and categorizing resources like GitHub repos, HuggingFace models and collections, and reading articles from various forums and threads."

This curation phase has led me to maintain a list of repositories I'm interested in contributing to, either by raising issues or solving existing ones.

## My First Significant Contribution

While I've made smaller contributions before, this particular contribution to Marimo stands out as my first significant fix for a user-reported issue. However, the path to a successful contribution wasn't straightforward.

### The Issue: ArviZ Plots Not Displaying

The problem was reported in [Issue #1033](https://github.com/marimo-team/marimo/issues/1033):

{{% callout warning %}}
"I am using the ArviZ library with PyMC and the plots are not being displayed. All I see is the axis info but not the plots."
{{% /callout %}}

### The First Attempt: A Learning Experience

My initial approach to solving this issue resulted in a Pull Request that, while well-intentioned, didn't quite hit the mark. Here's what I learned from this first attempt:

1. **Importance of Thorough Research**: I realized I needed to dive deeper into ArviZ's documentation and source code to truly understand the problem.
2. **Overcomplicating Solutions**: My first PR attempted to implement a complex formatter, which wasn't necessary for the issue at hand.

Here's an excerpt from my first PR comment:

> "This PR addresses issue #1033 by implementing an ArviZ formatter that can handle various plot types returned by ArviZ functions, including numpy arrays, matplotlib axes, and bokeh figures."

While this approach showed initiative, it wasn't the optimal solution for the problem at hand.

### The Transition: Life Gets in the Way

After my initial attempt, I found myself caught up with various commitments:

- College coursework intensified
- My ongoing capstone project demanded attention
- Other personal projects and responsibilities piled up

This period taught me an important lesson about balancing open-source contributions with other life commitments. It's okay to take a step back, regroup, and return to a problem with fresh eyes.

### Extensive Research and Testing

When I revisited the issue, I decided to conduct a thorough investigation of ArviZ's plotting capabilities. I created a comprehensive overview of ArviZ plot functions, their inputs, outputs, and behavior in different environments.

<details>
<summary>Click to view my detailed ArviZ plotting research</summary>

[Insert your detailed ArviZ plotting research here]

</details>

This extensive research was crucial in understanding the nuances of ArviZ's plotting functions and their interaction with Marimo's environment.

### The Solution: A Refined Approach

After my research, I proposed a simpler yet more effective solution:

<details>
<summary>Click to view the core logic of the solution</summary>

[Insert your core logic code here]

</details>

This approach focuses on:
1. Detecting ArviZ plot outputs
2. Handling various return types (numpy arrays, matplotlib Axes, etc.)
3. Ensuring consistent rendering across different ArviZ functions

### The Improved Pull Request

My second PR reflected a more mature and thoughtful approach to the problem. Here's an excerpt from the PR description:

> "This PR addresses the issue with ArviZ plots not displaying correctly in the Marimo output. It implements a custom formatter for ArviZ objects, specifically handling numpy arrays containing matplotlib `Axes` objects along with `az.InferenceData`."

The key improvements in this PR included:
- More targeted handling of ArviZ-specific outputs
- Better performance considerations
- Improved type checking and import handling

## Lessons Learned

This contribution taught me several valuable lessons:

1. **Deep Dive into Documentation**: Understanding the intricacies of ArviZ and matplotlib was crucial for finding the right solution.
2. **Iterative Problem-Solving**: My approach evolved from a simple fix to a comprehensive formatter implementation.
3. **Collaboration and Feedback**: The project maintainers provided invaluable guidance, helping me refine my solution.
4. **Attention to Detail**: Addressing issues like type checking, import statements, and performance considerations was essential for creating a robust solution.

## Looking Ahead

This experience has motivated me to continue contributing to open-source projects. I'm excited to tackle more complex issues and help improve tools that developers rely on every day.

{{% callout note %}}
Remember, every contribution, no matter how small, helps move the open-source community forward. Don't be afraid to dive in and start contributing!
{{% /callout %}}

I'm grateful for the opportunity to contribute to Marimo and look forward to many more open-source adventures ahead!

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

### ArviZ Plot Functions Overview

| Function | Input | Return | Behavior | Issues |
|----------|-------|--------|----------|--------|
| `plot_autocorr` | - | Axes or bokeh_figures | Causes typical issue error | Displays complex Axes structure |
| `plot_bf` | - | Dictionary, then plot | Plots without `plt.show()` | Returns text dictionary before plot |
| `plot_bpv` | - | 2D ndarray of Axes or Bokeh Figure | Plots without `plt.show()` | - |
| `plot_compare` | - | Axes or Bokeh Figure, pandas DataFrame | Issues warning | Not InferenceData |
| `plot_density` | - | 2D ndarray of Axes or Bokeh Figure | Causes typical issue error | Displays complex Axes structure |
| `plot_dist` | Array-like | Axes or Bokeh Figure | Plots without any issue | - |
| `plot_dist_comparison` | InferenceData | 2D ndarray of Axes | - | - |
| `plot_dot` | Array-like | Axes or Bokeh Figure | Plots without any issue | - |
| `plot_ecdf` | Array-like | Axes or Bokeh Figure | Plots without any issue | - |
| `plot_elpd` | Mapping of {str:ELPDData or InferenceData} | Axes or Bokeh Figure | - | - |
| `plot_energy` | obj | Axes or Bokeh Figure | Plots without any issue | - |
| `plot_ess` | InferenceData | Axes or Bokeh Figure | Causes typical issue error | - |
| `plot_forest` | InferenceData | 1D ndarray of Axes or Bokeh Figure | Plots without any issue | - |
| `plot_hdi` | Array-like | Axes or Bokeh Figure | Plots without any issue | - |
| `plot_kde` | Array-like | Axes or Bokeh Figure, optional glyphs list | Plots without any issue | - |
| `plot_khat` | ELPData or Array-like | Axes or Bokeh Figure | Plots without any issue | - |
| `plot_loo_pit` | InferenceData | Axes or Bokeh Figure | Plots without any issue | - |
| `plot_lm` | str or DataArray or ndarray | Axes or Bokeh Figure | Causes typical issue error | Issues with Bokeh backend |
| `plot_mcse` | InferenceData | Axes or Bokeh Figure | Causes typical issue error | Bokeh: Only axes, no data points |
| `plot_pair` | InferenceData | Axes or Bokeh Figure | Causes typical issue error | Works well with Bokeh |
| `plot_parallel` | InferenceData | Axes or Bokeh Figure | Plots without any issue | Bokeh: No controls in Marimo |
| `plot_posterior` | InferenceData | Axes or Bokeh Figure | Causes typical issue error | Bokeh: Incorrect rendering |
| `plot_ppc` | InferenceData | Axes or Bokeh Figure, optional Animation | Plots without any issue* | Bokeh doesn't work properly |
| `plot_rank` | InferenceData | Axes or Bokeh Figure | Causes typical issue error (sometimes) | - |
| `plot_separation` | InferenceData | Axes or Bokeh Figure | Plots without any issue | Trouble with Bokeh |
| `plot_trace` | InferenceData | Axes or Bokeh Figure | Causes typical issue error | Works well with Bokeh |
| `plot_ts` | InferenceData | Axes or Bokeh Figure | Causes typical issue error | No Bokeh support |
| `plot_violin` | InferenceData | Axes or Bokeh Figure | - | Works well with Bokeh |

### Common Issues and Observations

1. **Typical Issue Error**: Many functions require `plt.show()` at the end of the cell block to display the plot.

2. **Bokeh Backend Issues**:
   - Often opens a random new file in the temp folder
   - Controls for Bokeh don't always work correctly
   - Some functions work well with Bokeh, opening in a new window with proper controls
   - Others have rendering issues or don't display data correctly

3. **Plot Display**:
   - Some functions plot without requiring `plt.show()`
   - Others cause the "typical issue error" where `plt.show()` is needed

4. **Return Types**:
   - Most functions return matplotlib Axes or Bokeh Figures
   - Some return additional data structures (e.g., pandas DataFrames, dictionaries)

5. **Input Types**:
   - Many functions accept InferenceData objects
   - Some work with array-like inputs or specific data types (e.g., ELPDData)

6. **Specific Function Notes**:
   - `plot_autocorr` and `plot_density` return complex Axes structures
   - `plot_bf` returns a dictionary before displaying the plot
   - `plot_ppc` works fine for single plots but has issues with multiple plots using coords or flatten
   - `plot_parallel` may have text overlap issues with too much information

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

## Challenges and Learnings

Throughout this process, I faced several challenges that provided valuable learning experiences:

1. **CI/CD Pipeline Issues**: I repeatedly encountered failures in the repository's CI tests. This experience gave me practical insights into DevOps practices, complementing my theoretical knowledge from college courses.

2. **Code Style and Linting**: Adhering to the project's coding standards and passing linting checks taught me the importance of consistent code style in collaborative projects.

3. **Type Checking in Python**: Implementing proper type checking, especially for optional dependencies, was a new challenge that improved my understanding of Python's type system.

4. **Performance Considerations**: Optimizing the solution for large datasets without compromising functionality was an interesting problem to solve.

## Lessons Learned

This contribution journey taught me several valuable lessons:

1. **The Value of Persistence**: My initial failed attempt didn't discourage me but motivated me to learn more and come back stronger.

2. **The Importance of Thorough Research**: Deep diving into documentation and source code is crucial for understanding complex issues.

3. **Practical DevOps Experience**: Dealing with CI/CD pipelines and automated tests gave me hands-on experience that surpassed my college coursework.

4. **The Open Source Community**: The guidance and feedback from project maintainers were invaluable in refining my solution.

5. **Balancing Commitments**: Learning to manage open-source contributions alongside other responsibilities is a crucial skill.

## Looking Ahead

This experience has not only improved my technical skills but also given me a deeper appreciation for the open-source community. I'm excited to tackle more complex issues and continue contributing to projects that make a difference in the developer ecosystem.

{{% callout note %}}
Remember, the path to meaningful contributions isn't always straightforward. Embrace the learning process, be persistent, and don't be afraid to ask for help or take a step back when needed.
{{% /callout %}}

I'm grateful for the opportunity to contribute to Marimo and look forward to many more open-source adventures ahead!

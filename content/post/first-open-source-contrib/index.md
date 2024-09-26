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

### The Investigation

I dove deep into the ArviZ documentation and source code to understand the root cause. After extensive research, I discovered that adding `plt.show()` after any ArviZ plot function call resolved the display issue in most cases.

### The Solution

Instead of implementing a complex formatter, I proposed a simpler approach:

```python
import arviz as az
import matplotlib.pyplot as plt

# ArviZ plotting function
az.plot_trace(...)

# Add this line to display the plot
plt.show()
```

### The Pull Request

After further investigation and testing, I created a comprehensive Pull Request to address the issue:

1. Implemented a new `ArviZFormatter` class.
2. Added methods to handle numpy arrays containing matplotlib `Axes` objects.
3. Created utility methods to extract axes information and convert plots to HTML format.
4. Ensured consistent plot rendering across various ArviZ functions.

Here's a snippet of the core functionality:

```python
@staticmethod
def _contains_axes(arr: np.ndarray) -> bool:
    """
    Check if the numpy array contains any matplotlib Axes objects.
    To ensure performance, we limit the check to the first 100 items.
    """
    MAX_ITEMS_TO_CHECK = 100

    if arr.ndim == 1:
        return any(isinstance(item, plt.Axes) for item in arr[:MAX_ITEMS_TO_CHECK])
    elif arr.ndim == 2:
        items_checked = 0
        for row in arr:
            for item in row:
                if isinstance(item, plt.Axes):
                    return True
                items_checked += 1
                if items_checked >= MAX_ITEMS_TO_CHECK:
                    return False
    return False
```

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

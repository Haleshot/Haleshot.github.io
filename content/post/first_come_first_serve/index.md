---
title: First Come First Serve
subtitle: Exploring the First Come First Serve (FCFS) Scheduling Algorithm
summary: Implementation of the First Come First Serve (FCFS) algorithm for process scheduling, calculating turn around time, wait time, and their respective averages, with and without considering arrival times.
date: '2022-08-27T00:00:00Z'
lastmod: '2024-06-03T00:00:00Z'
draft: false
featured: true
image:
 caption: 'Program output'
 focal_point: ''
 placement: 2
 preview_only: false
authors:
 - admin
tags:
 - Operating Systems
 - Process Scheduling
 - FCFS
 - First Come First Serve
 - Algorithm Analysis
categories:
 - Programming
 - Algorithms
 - Computer Science
 - Academic
---

# First-Come-First-Serve (FCFS) Algorithm

The First-Come-First-Serve (FCFS) algorithm is the simplest scheduling algorithm in operating systems. It dispatches processes based on their arrival time, placing them on the ready queue. FCFS is a non-preemptive scheduling discipline, which means that once a process has the CPU, it will run to completion.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Calculating Average Waiting Time](#calculating-average-waiting-time)
- [Output](#output)

## Introduction
The FCFS scheduling algorithm is fair in terms of both formal and human sense of fairness. However, it has a drawback in that long jobs make short jobs wait, and unimportant jobs make important jobs wait. This algorithm is more predictable compared to most other schemes as it offers time.

## Features
- Simple and easy to understand code.
- Predictable behavior.
- Not suitable for scheduling interactive users as it cannot guarantee good response time.
- Often embedded within other scheduling schemes rather than used as a standalone master scheme in modern operating systems.

## Calculating Average Waiting Time
Average waiting time (AWT) is an important parameter to evaluate the performance of any scheduling algorithm. AWT represents the average waiting time of processes in the queue, waiting for the scheduler to select them for execution. The lower the average waiting time, the better the scheduling algorithm.

Consider the following processes with their respective burst times:

| Process | Burst Time |
|---------|------------|
| P1      | 24         |
| P2      | 3          |
| P3      | 3          |
| P4      | 6          |

Using the FCFS scheduling algorithm, we can calculate the average waiting time as follows:

```
Average Waiting Time = (0 + 24 + 27 + 30) / 4 = 20.25
```

Pictorial representation is:


![Pictorial Representation](https://user-images.githubusercontent.com/57552973/184399363-d5f003ce-8698-4e7e-bc81-c6eeb1d2abad.png)

## Output
The output of the `.py` file implementing the FCFS algorithm is shown below:

![FCFS Output](https://user-images.githubusercontent.com/57552973/187034305-6e0b4810-3da7-4f65-8fa4-a5be5488e97a.png)
![FCFS Output 2](https://user-images.githubusercontent.com/57552973/187034211-f5e90a8a-ff3c-4ea4-8f5d-eb01219821f7.png)

Please refer to the [FCFS Repository](https://github.com/Haleshot/OS-Programs/blob/master/First_Come_First_Serve/First_Come_First_Serve.py) for the complete code implementation.

For more information on scheduling algorithms and their analysis, please check the related files and code in this repository.

```python
# FCFS Algorithm with and without Arrival Time

# To Find - Turn Around Time and Wait Time and their respective average times

# import libraries
from tabulate import tabulate # For printing the result in a Tabular Format
```

```python
def sorting_arrival(l):
    return l[1] # Returns the Second element of the list which is Arrival Time
```

```python
def Turn_Around_Time(P, limit):
    # Declaring Variables for Calculating Total Turn Around Time
    total_tat = 0
    for i in range(limit):
        tat = P[i][3] - P[i][1]
        total_tat += tat # Formula For Turn Around Time -> Completion Time - Arrrival TIme
        P[i].append(tat) # Appending the Turn Around Time to the List

    avg_tat = total_tat/limit
    return avg_tat
```

```python
def Waiting_Time(P, limit):
    # Declaring Variables for Calculating Total Waiting Time
    total_wt = 0

    for i in range(limit):
        wt = P[i][4] - P[i][2]
        total_wt += wt # Formula For Waiting Time -> Turn Around Time - Burst TIme
        P[i].append(wt) # Appending the Waiting Time to the List

    avg_wt = total_wt/limit
    return avg_wt
```

```python
def Logic(P, limit):
    completion_time = 0
    exit_time = []
    
    for i in range(limit):
        if completion_time < P[i][1]:
            completion_time = P[i][1]
        completion_time += P[i][2]
        exit_time.append(completion_time)
        P[i].append(completion_time)
    
    tat = Turn_Around_Time(P, limit)
    wt = Waiting_Time(P, limit)

    P.sort(key=sorting_arrival) # Sorting the List by Arrivak Time
    headers = ["Process Number", "Arrival Time", "Burst Time", "Completion Time", "Turn Around Time", "Waiting Time"]
    print(tabulate(P, headers, tablefmt="psql"))

    # Printing the Average Waiting and Turn Around Time
    print("\nAverage Waiting Time is = ", round(wt, 2)) # Rounding off Average Waiting Time to 2 Decimal places
    print("Average Turn Around Time is = ", round(tat, 2)) # Rounding off Average Turn Around Time to 2 Decimal places
```

```python
# Main Function
def main():
    run = True
    while(run):

        # Declaring arrays
        processes = []

        print("\nMenu\nDo you want to assume : \n1. Arrival Time as 0\n2. Input Arrival Time\n3. Exit\n")
        ch = int(input("Enter Your Choice : "))

        if ch == 1:
            limit_process = int(input("Enter the Number of Processes : "))
            for i in range(limit_process):
                p = []
                arrival = 0
                burst = int(input("Enter the Burst Time for process {} : ".format(i)))
                process_id = "P" + str(i + 1)

                p.extend([process_id, arrival, burst])
                processes.append(p)

            Logic(processes, limit_process)
            run = int(input("\nWant to continue? (Yes = Input 1/false = Input 0) : "))

        elif ch == 2:
            limit_process = int(input("Enter the Number of Processes : "))
            for i in range(limit_process):
                p = []
                arrival = int(input("Enter the Arrival Time for process {} : ".format(i)))
                burst = int(input("Enter the Burst Time for process {} : ".format(i)))
                process_id = "P" + str(i + 1)

                p.extend([process_id, arrival, burst])
                processes.append(p)
                

            Logic(processes, limit_process)
            run = int(input("\nWant to continue? (Yes = Input 1/false = Input 0) : "))
        
        elif ch == 3:
            print("Thank You!")
            exit(0)

        else:
            print("Invalid Choice!")
            run = int(input("\nWant to continue? (Yes = Input 1/false = Input 0) : "))
```

```python
# Calling the Main Function
main()
```

    
    Menu
    Do you want to assume : 
    1. Arrival Time as 0
    2. Input Arrival Time
    3. Exit
    
    Enter Your Choice : 2
    Enter the Number of Processes : 4
    Enter the Arrival Time for process 0 : 0
    Enter the Burst Time for process 0 : 5
    Enter the Arrival Time for process 1 : 1
    Enter the Burst Time for process 1 : 3
    Enter the Arrival Time for process 2 : 2
    Enter the Burst Time for process 2 : 8
    Enter the Arrival Time for process 3 : 3
    Enter the Burst Time for process 3 : 6
    +------------------+----------------+--------------+-------------------+--------------------+----------------+
    | Process Number   |   Arrival Time |   Burst Time |   Completion Time |   Turn Around Time |   Waiting Time |
    |------------------+----------------+--------------+-------------------+--------------------+----------------|
    | P1               |              0 |            5 |                 5 |                  5 |              0 |
    | P2               |              1 |            3 |                 8 |                  7 |              4 |
    | P3               |              2 |            8 |                16 |                 14 |              6 |
    | P4               |              3 |            6 |                22 |                 19 |             13 |
    +------------------+----------------+--------------+-------------------+--------------------+----------------+
    
    Average Waiting Time is =  5.75
    Average Turn Around Time is =  11.25
    
    Want to continue? (Yes = Input 1/false = Input 0) : 0


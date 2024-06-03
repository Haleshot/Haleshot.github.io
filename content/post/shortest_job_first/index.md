---
title: Shortest Job First
subtitle: Implementing the Shortest Job First (SJF) Process Scheduling Algorithm
summary: Implementation of the Shortest Job First (SJF) algorithm for process scheduling, with and without considering arrival times (non-preemptive), calculating turn around time, wait time, and their respective averages.
date: '2022-08-12T00:00:00Z'
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
 - Shortest Job First
 - SJF
 - Non-Preemptive Scheduling
 - Algorithm Analysis
categories:
 - Programming
 - Algorithms
 - Computer Science
 - Academic
---

# Shortest Job First (SJF) Scheduling Algorithm

The Shortest Job First (SJF) scheduling algorithm is a non-preemptive scheduling algorithm that selects the process with the smallest burst time from the ready queue for execution. It aims to minimize the average waiting time among all scheduling algorithms. However, it may cause starvation if shorter processes keep arriving, which can be mitigated using the concept of aging. While it is practically infeasible for the operating system to know the exact burst times, SJF can be used in specialized environments where accurate estimates of running time are available.

## Table of Contents
- [Characteristics of SJF Scheduling](#characteristics-of-sjf-scheduling)
- [Algorithm](#algorithm)
- [Advantages](#advantages)
- [Disadvantages](#disadvantages)
- [Calculating Average Waiting Time](#calculating-average-waiting-time)
- [Example](#example)
- [Output](#output)

## Characteristics of SJF Scheduling
Here are the important characteristics of SJF Scheduling:
- Shortest Job First has the advantage of having a minimum average waiting time among all scheduling algorithms.
- It is a Greedy Algorithm.
- It may cause starvation if shorter processes keep coming. This problem can be solved using the concept of aging.
- It is practically infeasible as the operating system may not know the burst times and therefore may not sort them. However, there are methods to estimate the execution time for a job based on previous execution times.
- SJF can be used in specialized environments where accurate estimates of running time are available.

## Algorithm
The SJF scheduling algorithm follows these steps:
1. Sort all the processes according to their arrival time.
2. Select the process with the minimum arrival time and minimum burst time.
3. After completing the selected process, create a pool of processes that arrive afterward until the completion of the previous process. Select the process with the minimum burst time from this pool.

## Advantages
- Reduces the average waiting time significantly.
- Better compared to First-Come-First-Serve (FCFS) scheduling.
- Helps in achieving optimal turnaround time.

## Disadvantages
- It could cause starvation in some cases, which can be solved by using aging techniques.
- The execution time of a process is a prerequisite, which is hard to predict.
- Cannot be used for short-term CPU scheduling since predicting the CPU burst time is challenging.

## Calculating Average Waiting Time
Average waiting time (AWT) is a crucial parameter to evaluate the performance of any scheduling algorithm. AWT represents the average waiting time of processes in the queue, waiting for the scheduler to select them for execution.

## Example
Consider the following example with five jobs (P1, P2, P3, P4, and P5) having their arrival time and burst time given:

![Example Processes 1](https://user-images.githubusercontent.com/57552973/184400549-31fb2a38-12d9-434a-9b1a-0038bfdd2bc6.png)

![Example Processes 2](https://user-images.githubusercontent.com/57552973/184400622-95619498-fec1-44d1-82a9-0e9f1c61242f.png)

## Output
The output of the `.py` file implementing the SJF scheduling algorithm is shown below:

![SJF Output 1](https://user-images.githubusercontent.com/57552973/184400016-9f123361-077e-43fb-a235-6b060f9ffea5.png)

![SJF Output 2](https://user-images.githubusercontent.com/57552973/184400117-82cfa672-932a-42f2-b524-fd9121506d84.png)

Please refer to the [SJF Repository](https://github.com/Haleshot/OS-Programs/blob/master/Shortest_Job_First/Shortest_Job_First.py) for the complete code implementation.

For more information on scheduling algorithms and their analysis, please check the related files and code in this repository.

```python
# SJF Algorithm with and without Arrival Time (Non Preemptive)

# To Find - Turn Around Time and Wait Time and their respective average times

# import libraries
from tabulate import tabulate # For printing the result in a Tabular Format
```

```python
# Functions to sort the list which contains Arrival and Burst Times according to Burst Time
def sorting_burst(l):
    return l[2] # Returns the Third element of the list which is Burst Time
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
        tat = P[i][4] - P[i][1]
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
        wt = P[i][5] - P[i][2]
        total_wt += wt # Formula For Waiting Time -> Turn Around Time - Burst TIme
        P[i].append(wt) # Appending the Waiting Time to the List

    avg_wt = total_wt/limit
    return avg_wt
```

```python
def Logic(P, limit):
    execution_time = []
    exit_time = [] # To note the completion time of a process -> the end time of previous process + burst time of current process
    completion_time = 0 # Execution Time for a process

    # Sorting Processes by Arrival Time
    P.sort(key=sorting_arrival)

    for i in range(limit):
        buffer = []
        not_arrived = [] # For processes which have not yet arrived
        arrived = [] # For processes which have arrived

        for j in range(limit):
            if (P[j][1] <= completion_time and P[j][3] == 0): # Checking whether the arrival time of the process is less than Completion time or not
                buffer.extend([P[j][0], P[j][1], P[j][2]])
                arrived.append(buffer)
                buffer = []

            elif (P[j][3]  == 0): # Checking whether the process has been executed or not
                buffer.extend([P[j][0], P[j][1], P[j][2]])
                not_arrived.append(buffer)
                buffer = []

        if (len(arrived)) != 0:

            arrived.sort(key=sorting_burst) # Sorting Processes by Burst Time 
            execution_time.append(completion_time)

            completion_time += arrived[0][2]
            exit_time.append(completion_time)

            for k in range(limit):
                if P[k][0] == arrived[0][0]:
                    break
            
            P[k][3] = 1
            P[k].append(completion_time)

        elif (len(arrived)) == 0:
            if completion_time < not_arrived[0][1]:
                completion_time = not_arrived[0][1]

            execution_time.append(completion_time)

            completion_time += not_arrived[0][2]
            exit_time.append(completion_time)

            for k in range(limit):
                if P[k][0] == not_arrived[0][0]:
                    break
            
            P[k][3] = 1
            P[k].append(completion_time)

    tat = Turn_Around_Time(P, limit)
    wt = Waiting_Time(P, limit)

    P.sort(key=sorting_burst) # Sorting the List by Burst Time (Order in which processes are executed)
    headers = ["Process Number", "Arrival Time", "Burst Time", "Completed Status", "Total Execution Time", "Turn Around Time", "Completion Time"]
    print(tabulate(P, headers, tablefmt="psql"))

    # Printing the Average Waiting and Turn Around Time
    print("\nAverage Waiting Time is = ", round(wt, 2)) # Rounding off Average Waiting Time to 2 Decimal places
    print("Average Turn Around Time is = ", round(tat, 2)) # Rounding off Average Turn Around Time to 2 Decimal places
```

```python
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

                p.extend([process_id, arrival, burst, 0]) # Forming a list of info entered by the user
                processes.append(p)

            Logic(processes , limit_process)
            run = int(input("\nWant to continue? (Yes = Input 1/false = Input 0) : "))

        elif ch == 2:
            limit_process = int(input("Enter the Number of Processes : "))
            for i in range(limit_process):
                p = []
                arrival = int(input("Enter the Arrival Time for process {} : ".format(i)))
                burst = int(input("Enter the Burst Time for process {} : ".format(i)))
                process_id = "P" + str(i + 1)

                p.extend([process_id, arrival, burst, 0])
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
main()
```

    
    Menu
    Do you want to assume : 
    1. Arrival Time as 0
    2. Input Arrival Time
    3. Exit
    
    Enter Your Choice : 1
    Enter the Number of Processes : 3
    Enter the Burst Time for process 0 : 5
    Enter the Burst Time for process 1 : 4
    Enter the Burst Time for process 2 : 6
    +------------------+----------------+--------------+--------------------+------------------------+--------------------+-------------------+
    | Process Number   |   Arrival Time |   Burst Time |   Completed Status |   Total Execution Time |   Turn Around Time |   Completion Time |
    |------------------+----------------+--------------+--------------------+------------------------+--------------------+-------------------|
    | P2               |              0 |            4 |                  1 |                      4 |                  4 |                 0 |
    | P1               |              0 |            5 |                  1 |                      9 |                  9 |                 4 |
    | P3               |              0 |            6 |                  1 |                     15 |                 15 |                 9 |
    +------------------+----------------+--------------+--------------------+------------------------+--------------------+-------------------+
    
    Average Waiting Time is =  4.33
    Average Turn Around Time is =  9.33
    
    Want to continue? (Yes = Input 1/false = Input 0) : 0


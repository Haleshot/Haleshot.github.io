---
title: Priority Scheduling
date: '2024-06-03'
---
```python
# Priority Scheduling Algorithm with and without Arrival Time (Preemptive)

# To Find - Turn Around Time and Wait Time and their respective average times
# import libraries
from tabulate import tabulate # For printing the result in a Tabular Format
```

```python
# Functions to sort the list according to the Priority Time
def sorting_priority(l):
    return l[3] # Returns the Third element of the list which is Priority Time
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
        tat = P[i][6] - P[i][1]
        total_tat += tat # Formula For Turn Around Time -> Completion Time - Arrival TIme
        P[i].append(tat) # Appending the Turn Around Time to the List

    avg_tat = total_tat/limit
    return avg_tat
```

```python
def Waiting_Time(P, limit):
    # Declaring Variables for Calculating Total Waiting Time
    total_wt = 0

    for i in range(limit):
        wt = P[i][6] - P[i][2]
        total_wt += wt # Formula For Waiting Time -> Turn Around Time - Burst Time
        P[i].append(wt) # Appending the Waiting Time to the List

    avg_wt = total_wt/limit
    return avg_wt
```

```python
def Logic(P, limit):
    
    completion_time = 0 # Execution Time for a process
    exit_time = [] # To note the completion time of a process -> the end time of previous process + burst time of current process

    # Sorting Processes by Arrival Time
    P.sort(key=sorting_arrival)

    while True: # The loop runs till all the processes have been executed successfully
        arrived = []  # Contains Processes which have completed their respective execution
        not_arrived = [] # Contains Processes which have not completed their respective execution
        buffer = []
        for i in range(limit):
            if(P[i][1] <= completion_time and P[i][4] == 0): # Checking whether the arrival time of the process is less
                    # than Completion time or not and if the process has not been executed
                buffer.extend([P[i][0], P[i][1], P[i][2], P[i][3], P[i][4], P[i][5]])
                arrived.append(buffer) # Appending the process to the arrived queue
                buffer = []
            elif (P[i][4] == 0): # Only checking whether process has been executed or not
                buffer.extend([P[i][0], P[i][1], P[i][2], P[i][3], P[i][4], P[i][5]])
                not_arrived .append(buffer)
                buffer = []

        if len(arrived) == 0 and len(not_arrived) == 0:
            break
        if len(arrived) != 0:
            arrived.sort(key=sorting_priority, reverse=True)
            completion_time += 1
            exit_time.append(completion_time)
            for i in range(limit):
                if(P[i][0] == arrived[0][0]):
                    break
            P[i][2] -= 1
            if P[i][2] == 0: # Checking whether the Process has been executed till its Burst Time
                P[i][4] = 1
                P[i].append(completion_time)
            
        if len(arrived) == 0:
            not_arrived.sort(key=sorting_arrival)
            if completion_time < not_arrived[0][1]:
                completion_time = not_arrived[0][1]

            completion_time += 1
            exit_time.append(completion_time)
            for i in range(limit):
                if(P[i][0] == not_arrived[0][0]):
                    break
            P[i][2] -= 1
            if P[i][2] == 0: # Checking whether the Process has been executed till its Burst Time
                P[i][4] = 1
                P[i].append(completion_time)

    tat = Turn_Around_Time(P, limit)
    wt = Waiting_Time(P, limit)

    P.sort(key=sorting_priority) # Sorting the List by Priority Time (Order in which processes are executed)
    headers = ["Process Number", "Arrival Time", "Remainder Burst Time", "Priority", "Completed Status", "Original Burst Time", "Total Execution Time", "Turn Around Time", "Waiting Time"]
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
                priority = int(input("Enter the Priority Number for process {} : ".format(i)))
                p.extend([process_id, arrival, burst, priority, 0, burst]) # Forming a list of info entered by the user, 
                # 0 is for completion status
                processes.append(p)

            Logic(processes , limit_process)
            run = int(input("\nWant to continue? (Yes = Input 1/false = Input 0) : "))

        elif ch == 2:
            limit_process = int(input("Enter the Number of Processes : "))
            for i in range(limit_process):
                p = []
                arrival = int(input("Enter the Arrival Time for process {} : ".format(i)))
                burst = int(input("Enter the Burst Time for process {} : ".format(i)))
                priority = int(input("Enter the Priority Number for process {} : ".format(i)))
                process_id = "P" + str(i + 1)

                p.extend([process_id, arrival, burst, priority, 0, burst])
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
    Enter the Number of Processes : 5
    Enter the Arrival Time for process 0 : 0
    Enter the Burst Time for process 0 : 3
    Enter the Priority Number for process 0 : 3
    Enter the Arrival Time for process 1 : 1
    Enter the Burst Time for process 1 : 6
    Enter the Priority Number for process 1 : 4
    Enter the Arrival Time for process 2 : 3
    Enter the Burst Time for process 2 : 1
    Enter the Priority Number for process 2 : 9
    Enter the Arrival Time for process 3 : 2
    Enter the Burst Time for process 3 : 2
    Enter the Priority Number for process 3 : 7
    Enter the Arrival Time for process 4 : 4
    Enter the Burst Time for process 4 : 4
    Enter the Priority Number for process 4 : 8
    +------------------+----------------+------------------------+------------+--------------------+-----------------------+------------------------+--------------------+----------------+
    | Process Number   |   Arrival Time |   Remainder Burst Time |   Priority |   Completed Status |   Original Burst Time |   Total Execution Time |   Turn Around Time |   Waiting Time |
    |------------------+----------------+------------------------+------------+--------------------+-----------------------+------------------------+--------------------+----------------|
    | P1               |              0 |                      0 |          3 |                  1 |                     3 |                     16 |                 16 |             16 |
    | P2               |              1 |                      0 |          4 |                  1 |                     6 |                     14 |                 13 |             14 |
    | P4               |              2 |                      0 |          7 |                  1 |                     2 |                      9 |                  7 |              9 |
    | P5               |              4 |                      0 |          8 |                  1 |                     4 |                      8 |                  4 |              8 |
    | P3               |              3 |                      0 |          9 |                  1 |                     1 |                      4 |                  1 |              4 |
    +------------------+----------------+------------------------+------------+--------------------+-----------------------+------------------------+--------------------+----------------+
    
    Average Waiting Time is =  10.2
    Average Turn Around Time is =  8.2


import csv
import heapq
import statistics

INPUT_FILE = "supermarket_queue_dataset.csv"
NUM_CASHIERS = 2

customers = []
with open(INPUT_FILE, newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    for row in reader:
        customers.append({
            "Customer_ID": row["Customer_ID"],
            "Arrival_Time_Min": float(row["Arrival_Time_Min"]),
            "Number_of_Items": int(row["Number_of_Items"]),
            "Service_Time_Min": float(row["Estimated_Service_Time_Min"])
        })

# Each cashier is stored as (next_available_time, cashier_id)
cashiers = [(0.0, i + 1) for i in range(NUM_CASHIERS)]
heapq.heapify(cashiers)

results = []
for customer in customers:
    available_time, cashier_id = heapq.heappop(cashiers)

    arrival = customer["Arrival_Time_Min"]
    service_start = max(arrival, available_time)
    waiting_time = service_start - arrival
    departure_time = service_start + customer["Service_Time_Min"]
    time_in_system = departure_time - arrival

    results.append({
        **customer,
        "Cashier_ID": cashier_id,
        "Service_Start_Min": round(service_start, 2),
        "Waiting_Time_Min": round(waiting_time, 2),
        "Departure_Time_Min": round(departure_time, 2),
        "Time_In_System_Min": round(time_in_system, 2)
    })

    heapq.heappush(cashiers, (departure_time, cashier_id))

total_service_time = sum(r["Service_Time_Min"] for r in results)
total_observation_time = max(r["Departure_Time_Min"] for r in results)
average_waiting_time = statistics.mean(r["Waiting_Time_Min"] for r in results)
maximum_waiting_time = max(r["Waiting_Time_Min"] for r in results)
throughput = len(results) / total_observation_time
utilization = total_service_time / (NUM_CASHIERS * total_observation_time)

print("SUPERMARKET CHECKOUT QUEUE PERFORMANCE RESULTS")
print("-" * 50)
print(f"Customers processed: {len(results)}")
print(f"Number of cashiers: {NUM_CASHIERS}")
print(f"Average waiting time: {average_waiting_time:.2f} minutes")
print(f"Maximum waiting time: {maximum_waiting_time:.2f} minutes")
print(f"Throughput: {throughput:.2f} customers/minute")
print(f"Cashier utilization: {utilization * 100:.2f}%")

with open("simulation_results.csv", "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print("\nDetailed results saved to simulation_results.csv")

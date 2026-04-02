import pygad
import numpy as np

def fitness_func(ga_instance, solution, solution_idx):
    # Lấy ma trận Q trực tiếp từ đối tượng ga_instance
    # Điều này đảm bảo mọi tiến trình con đều có dữ liệu
    Q = ga_instance.Q_matrix
    
    # Tính toán Energy: x^T * Q * x
    energy = solution @ Q @ solution
    return -energy

def run_ga_optimization(Q, num_generations=5000, sol_per_pop=500):
    num_vars = Q.shape[0]

    ga_instance = pygad.GA(
        num_generations=num_generations,
        num_parents_mating=int(sol_per_pop/5),
        fitness_func=fitness_func,
        sol_per_pop=sol_per_pop,
        num_genes=num_vars,
        gene_space=[0, 1],
        
        # Chạy đa nhân 16 core
        parallel_processing=["thread", 20], 
        
        parent_selection_type="tournament",
        K_tournament=3,
        crossover_type="two_points",
        mutation_type="random",
        mutation_percent_genes=10,
        stop_criteria=["saturate_100"],
        keep_parents=10,
        save_best_solutions=False
    )

    # MẸO QUAN TRỌNG: Gán ma trận Q vào một thuộc tính mới của ga_instance
    # PyGAD sẽ copy thuộc tính này sang các tiến trình con cho Dũng
    ga_instance.Q_matrix = Q

    ga_instance.run()
    
    solution, solution_fitness, _ = ga_instance.best_solution()
    return solution, -solution_fitness
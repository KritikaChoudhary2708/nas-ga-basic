import torch, sys, os, pickle
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10

from model_ga import GeneticAlgorithm
from model_cnn import CNN

# if __name__ == "__main__":

parent = os.path.abspath('')
if not os.path.exists(os.path.join(parent, 'outputs')):
    os.mkdir(os.path.join(parent, 'outputs'))
all_logs = [i for i in os.listdir(os.path.join(parent, 'outputs')) if 'log' in i]
os.mkdir(os.path.join(parent, 'outputs', f'run_{len(all_logs)+1}'))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sys.stdout = open(os.path.join(parent, 'outputs', f'run_{len(all_logs)+1}', f'nas_run.log'), 'w')

print(f"Using device: {device}", flush=True)

# Load CIFAR-10 dataset (reduced for faster NAS)
transform = T.Compose([
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = CIFAR10(root='./data', train=True, download=True, transform=transform)
valset = CIFAR10(root='./data', train=False, download=True, transform=transform)

# Use only 3000 samples for faster NAS
train_subset = Subset(trainset, range(3000)) #original train set size was 50000 but system was running slow so reduced it to 3000
val_subset = Subset(valset, range(500)) #original val set size was 10000 but system was running slow so reduced it to 500

train_loader = DataLoader(train_subset, batch_size=512, shuffle=True) #original batch size was 128 but system was running slow so increased it to 512
val_loader = DataLoader(val_subset, batch_size=512, shuffle=False) #original batch size was 128 but system was running slow so increased it to 512  

# Run NAS with GA
ga = GeneticAlgorithm(
    population_size=6,  # Reduced from 10 for faster runtime
    generations=3,       # Reduced from 5 for faster runtime
    mutation_rate=0.3,
    crossover_rate=0.7
)

best_arch = ga.evolve(train_loader, val_loader, device, run=len(all_logs)+1)

print(f"\n{'='*60}", flush=True)
print("FINAL BEST ARCHITECTURE", flush=True)
print(f"{'='*60}", flush=True)
print(f"Genes: {best_arch.genes}", flush=True)
print(f"Accuracy: {best_arch.accuracy:.4f}", flush=True)
print(f"Fitness: {best_arch.fitness:.4f}", flush=True)

# Build and test final model
final_model = CNN(best_arch.genes).to(device)
print(f"\nTotal parameters: {sum(p.numel() for p in final_model.parameters()):,}", flush=True)
print(f"\nModel architecture:\n{final_model}", flush=True)

with open(os.path.join(parent, 'outputs', f'run_{len(all_logs)+1}', f"best_arch.pkl"), 'wb') as f:
    pickle.dump(best_arch, f)

sys.stdout = sys.__stdout__
import os
import time
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
import psutil
import gc
from datetime import datetime

class LLMPerformanceMetrics:
    def __init__(self, model_name="deepseek-ai/deepseek-llm-7b-chat"):
        self.console = Console()
        self.console.print("[bold blue]Initializing Performance Metrics Tool...[/bold blue]")
        
        # Configure 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        
        # Load model and tokenizer
        self.console.print(f"[yellow]Loading model: {model_name} for benchmarking...[/yellow]")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Performance metrics storage
        self.metrics_history = []
        self.current_benchmark = {}
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.console.print(f"[green]Using GPU: {torch.cuda.get_device_name(0)}[/green]")
        else:
            self.device = torch.device("cpu")
            self.console.print("[yellow]No GPU detected, using CPU[/yellow]")
        
        self.console.print("[bold green]Performance Metrics Tool initialized![/bold green]")
    
    def generate_with_metrics(self, prompt, max_new_tokens=512, temperature=0.7, top_p=0.9):
        """Generate text and collect performance metrics"""
        # Tokenize input
        input_ids = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_token_count = input_ids.input_ids.shape[1]
        
        # Track memory before generation
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            start_mem = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
        
        # Generate with timing
        start_time = time.time()
        
        with torch.no_grad():
            outputs = self.model.generate(
                **input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Calculate time and token metrics
        end_time = time.time()
        generation_time = end_time - start_time
        
        # Extract only the new tokens
        output_token_count = outputs[0].shape[0] - input_token_count
        tokens_per_second = output_token_count / generation_time
        
        # Track peak memory
        if torch.cuda.is_available():
            end_mem = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
            peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB
            memory_used = peak_mem - start_mem
        else:
            memory_used = None
            peak_mem = None
            end_mem = None
        
        # Decode output
        generated_text = self.tokenizer.decode(outputs[0][input_token_count:], skip_special_tokens=True)
        
        # Collect metrics
        metrics = {
            "timestamp": datetime.now(),
            "input_tokens": input_token_count,
            "output_tokens": output_token_count,
            "generation_time": generation_time,
            "tokens_per_second": tokens_per_second,
            "memory_start_gb": start_mem if torch.cuda.is_available() else None,
            "memory_peak_gb": peak_mem if torch.cuda.is_available() else None,
            "memory_used_gb": memory_used if torch.cuda.is_available() else None,
        }
        
        self.metrics_history.append(metrics)
        return generated_text, metrics
    
    def run_benchmark(self, prompt_length_tokens=[100, 200, 500, 1000], 
                      output_tokens=[128, 256, 512, 1024], 
                      runs_per_config=3):
        """Run a comprehensive benchmark with different input/output sizes"""
        self.console.print("[bold]Starting comprehensive benchmark...[/bold]")
        
        all_metrics = []
        benchmark_results = {}
        
        # Create sample prompts of different lengths
        base_prompt = "Explain the concept of artificial intelligence and its impact on society. " * 100
        prompts = {}
        
        for length in prompt_length_tokens:
            # Create a prompt of approximately the desired token length
            estimated_chars = length * 4  # rough estimate of chars per token
            prompt = base_prompt[:estimated_chars]
            tokens = self.tokenizer(prompt, return_tensors="pt")
            actual_length = tokens.input_ids.shape[1]
            prompts[length] = {
                "text": prompt,
                "actual_tokens": actual_length
            }
        
        # Run benchmark for each configuration
        total_configs = len(prompt_length_tokens) * len(output_tokens) * runs_per_config
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task = progress.add_task("[cyan]Running benchmark...", total=total_configs)
            
            for prompt_size in prompt_length_tokens:
                prompt = prompts[prompt_size]["text"]
                actual_input_size = prompts[prompt_size]["actual_tokens"]
                
                for output_size in output_tokens:
                    config_metrics = []
                    
                    for run in range(runs_per_config):
                        progress.update(task, description=f"Testing {actual_input_size} → {output_size} tokens (Run {run+1}/{runs_per_config})")
                        
                        # Clear cache between runs
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()
                        
                        # Run generation
                        _, metrics = self.generate_with_metrics(
                            prompt=prompt,
                            max_new_tokens=output_size
                        )
                        
                        config_metrics.append(metrics)
                        all_metrics.append(metrics)
                        progress.update(task, advance=1)
                    
                    # Store average metrics for this configuration
                    avg_tokens_per_sec = sum(m["tokens_per_second"] for m in config_metrics) / len(config_metrics)
                    benchmark_results[f"in{actual_input_size}_out{output_size}"] = {
                        "input_tokens": actual_input_size,
                        "output_tokens": output_size,
                        "avg_tokens_per_second": avg_tokens_per_sec,
                        "runs": config_metrics
                    }
        
        self.current_benchmark = {
            "timestamp": datetime.now(),
            "configurations": benchmark_results,
            "all_metrics": all_metrics
        }
        
        # Display results
        self.display_benchmark_results()
        return self.current_benchmark
    
    def display_benchmark_results(self):
        """Display the results of the benchmark in a table"""
        if not self.current_benchmark:
            self.console.print("[red]No benchmark results available.[/red]")
            return
        
        # Create a nice table
        table = Table(title="LLM Inference Performance Metrics")
        table.add_column("Input Tokens", justify="right")
        table.add_column("Output Tokens", justify="right")
        table.add_column("Tokens/Second", justify="right")
        table.add_column("Generation Time (s)", justify="right")
        if torch.cuda.is_available():
            table.add_column("Memory Used (GB)", justify="right")
        
        # Add data rows
        for config_key, config_data in self.current_benchmark["configurations"].items():
            input_tokens = config_data["input_tokens"]
            output_tokens = config_data["output_tokens"]
            avg_tokens_per_second = config_data["avg_tokens_per_second"]
            avg_time = sum(m["generation_time"] for m in config_data["runs"]) / len(config_data["runs"])
            
            if torch.cuda.is_available():
                avg_memory = sum(m["memory_used_gb"] for m in config_data["runs"] if m["memory_used_gb"]) / len(config_data["runs"])
                table.add_row(
                    str(input_tokens),
                    str(output_tokens),
                    f"{avg_tokens_per_second:.2f}",
                    f"{avg_time:.2f}",
                    f"{avg_memory:.2f}"
                )
            else:
                table.add_row(
                    str(input_tokens),
                    str(output_tokens),
                    f"{avg_tokens_per_second:.2f}",
                    f"{avg_time:.2f}"
                )
        
        self.console.print(table)
    
    def plot_metrics(self, save_path=None):
        """Plot the performance metrics"""
        if not self.metrics_history:
            self.console.print("[red]No metrics available to plot.[/red]")
            return
        
        plt.figure(figsize=(12, 8))
        
        # Plot 1: Tokens per second by output length
        plt.subplot(2, 1, 1)
        x_output_tokens = [m["output_tokens"] for m in self.metrics_history]
        y_tokens_per_second = [m["tokens_per_second"] for m in self.metrics_history]
        
        plt.scatter(x_output_tokens, y_tokens_per_second)
        plt.plot(np.unique(x_output_tokens), 
                 np.poly1d(np.polyfit(x_output_tokens, y_tokens_per_second, 1))(np.unique(x_output_tokens)), 
                 color='red')
        plt.xlabel("Output Tokens")
        plt.ylabel("Tokens per Second")
        plt.title("Inference Speed vs Output Length")
        plt.grid(True)
        
        # Plot 2: Generation Time vs Output Length
        plt.subplot(2, 1, 2)
        x_output_tokens = [m["output_tokens"] for m in self.metrics_history]
        y_generation_time = [m["generation_time"] for m in self.metrics_history]
        
        plt.scatter(x_output_tokens, y_generation_time)
        plt.plot(np.unique(x_output_tokens), 
                 np.poly1d(np.polyfit(x_output_tokens, y_generation_time, 1))(np.unique(x_output_tokens)), 
                 color='red')
        plt.xlabel("Output Tokens")
        plt.ylabel("Generation Time (s)")
        plt.title("Generation Time vs Output Length")
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            self.console.print(f"[green]Plot saved to {save_path}[/green]")
        
        plt.show()
    
    def save_metrics_to_file(self, filename="llm_metrics.csv"):
        """Save metrics to a CSV file"""
        if not self.metrics_history:
            self.console.print("[red]No metrics to save.[/red]")
            return
        
        import pandas as pd
        df = pd.DataFrame(self.metrics_history)
        df.to_csv(filename, index=False)
        self.console.print(f"[green]Metrics saved to {filename}[/green]")
        
    def run_interactive(self):
        """Run an interactive performance testing session"""
        self.console.print("[bold purple]LLM Performance Metrics Tool[/bold purple]")
        self.console.print("Type '/help' for available commands.")
        
        while True:
            command = input("\nCommand> ")
            
            if command.lower() in ['/exit', '/quit']:
                self.console.print("[bold red]Exiting...[/bold red]")
                break
                
            elif command.lower() == '/help':
                self.console.print("""
                ## Available Commands:
                - `/benchmark`: Run comprehensive benchmark
                - `/test <prompt>`: Test with a specific prompt
                - `/plot`: Plot collected metrics
                - `/save [filename]`: Save metrics to CSV
                - `/stats`: Show statistics of collected metrics
                - `/exit` or `/quit`: Exit the tool
                """)
                
            elif command.lower() == '/benchmark':
                self.run_benchmark()
                
            elif command.lower().startswith('/test '):
                prompt = command[6:].strip()
                self.console.print("[cyan]Generating with metrics...[/cyan]")
                _, metrics = self.generate_with_metrics(prompt)
                
                # Display metrics
                self.console.print("\n[bold]Performance Metrics:[/bold]")
                self.console.print(f"Input Tokens: {metrics['input_tokens']}")
                self.console.print(f"Output Tokens: {metrics['output_tokens']}")
                self.console.print(f"Generation Time: {metrics['generation_time']:.2f}s")
                self.console.print(f"[bold green]Tokens/Second: {metrics['tokens_per_second']:.2f}[/bold green]")
                
                if torch.cuda.is_available():
                    self.console.print(f"GPU Memory Used: {metrics['memory_used_gb']:.2f} GB")
                
            elif command.lower() == '/plot':
                self.plot_metrics()
                
            elif command.lower().startswith('/save'):
                parts = command.split(maxsplit=1)
                filename = parts[1] if len(parts) > 1 else "llm_metrics.csv"
                self.save_metrics_to_file(filename)
                
            elif command.lower() == '/stats':
                if not self.metrics_history:
                    self.console.print("[yellow]No metrics collected yet.[/yellow]")
                    continue
                    
                # Calculate statistics
                tokens_per_second = [m["tokens_per_second"] for m in self.metrics_history]
                avg_tps = sum(tokens_per_second) / len(tokens_per_second)
                max_tps = max(tokens_per_second)
                min_tps = min(tokens_per_second)
                
                self.console.print("\n[bold]Performance Statistics:[/bold]")
                self.console.print(f"Samples: {len(self.metrics_history)}")
                self.console.print(f"Average Tokens/Second: {avg_tps:.2f}")
                self.console.print(f"Maximum Tokens/Second: {max_tps:.2f}")
                self.console.print(f"Minimum Tokens/Second: {min_tps:.2f}")
                
                if torch.cuda.is_available():
                    memory_usage = [m["memory_used_gb"] for m in self.metrics_history if m["memory_used_gb"]]
                    if memory_usage:
                        avg_mem = sum(memory_usage) / len(memory_usage)
                        max_mem = max(memory_usage)
                        self.console.print(f"Average Memory Usage: {avg_mem:.2f} GB")
                        self.console.print(f"Peak Memory Usage: {max_mem:.2f} GB")
            
            else:
                self.console.print("[red]Unknown command. Type '/help' for available commands.[/red]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Performance Metrics Tool")
    parser.add_argument("--model", type=str, default="deepseek-ai/deepseek-llm-7b-chat", 
                        help="Model to benchmark")
    parser.add_argument("--benchmark", action="store_true", 
                        help="Run benchmark automatically at startup")
    parser.add_argument("--save", type=str, 
                        help="Save metrics to the specified CSV file after benchmark")
    parser.add_argument("--plot", type=str, 
                        help="Save plot to the specified image file after benchmark")
    
    args = parser.parse_args()
    
    # Initialize metrics tool
    metrics_tool = LLMPerformanceMetrics(model_name=args.model)
    
    # Run benchmark if requested
    if args.benchmark:
        metrics_tool.run_benchmark()
        
        if args.plot:
            metrics_tool.plot_metrics(save_path=args.plot)
            
        if args.save:
            metrics_tool.save_metrics_to_file(filename=args.save)
    else:
        metrics_tool.run_interactive()
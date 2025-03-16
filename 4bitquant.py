import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import time
import argparse
from rich.console import Console
from rich.markdown import Markdown

#tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")

class DeepSeekChatBot:
    def __init__(self, model_name="deepseek-ai/deepseek-llm-7b-chat", temperature=0.7, top_p=0.9, max_tokens=512):
        self.console = Console()
        self.console.print("[bold blue]Initializing DeepSeek Chatbot...[/bold blue]")
        
        # Configure 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        
        # Load model and tokenizer
        self.console.print(f"[yellow]Loading model: {model_name}...[/yellow]")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set generation parameters
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        
        # Initialize conversation history
        self.conversation_history = []
        self.system_prompt = "You are a helpful AI assistant that provides accurate and thoughtful answers."
        
        self.console.print("[bold green]Chatbot initialized successfully![/bold green]")
    
    def format_prompt(self):
        """Format conversation history into a proper prompt for DeepSeek model"""
        messages = [{"role": "system", "content": self.system_prompt}]
        
        for message in self.conversation_history:
            messages.append(message)
            
        # DeepSeek models use a specific chat template
        prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        return prompt
    
    def generate_response(self, user_input):
        # Add user message to history
        self.conversation_history.append({"role": "user", "content": user_input})
        
        # Format full conversation with history
        prompt = self.format_prompt()
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Generate response
        start_time = time.time()
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Extract only the new tokens
        new_tokens = outputs[0][inputs.input_ids.shape[1]:]
        response_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        # Add assistant response to history
        self.conversation_history.append({"role": "assistant", "content": response_text})
        
        # Calculate response time
        elapsed_time = time.time() - start_time
        
        return response_text, elapsed_time
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        return "Conversation history cleared."
    
    def set_system_prompt(self, new_prompt):
        """Update the system prompt"""
        self.system_prompt = new_prompt
        return f"System prompt updated to: {new_prompt}"
    
    def run(self):
        self.console.print("[bold purple]DeepSeek Chatbot is ready! Type '/help' to see available commands.[/bold purple]")
        
        while True:
            user_input = input("\n[You]: ")
            
            # Handle commands
            if user_input.lower() in ['/exit', '/quit']:
                self.console.print("[bold red]Exiting chatbot...[/bold red]")
                break
                
            elif user_input.lower() == '/help':
                self.console.print(Markdown("""
                ## Available Commands:
                - `/exit` or `/quit`: Exit the chatbot
                - `/clear`: Clear conversation history
                - `/system <prompt>`: Set a new system prompt
                - `/temp <value>`: Set temperature (0.0-1.0)
                - `/tokens <value>`: Set max tokens
                - `/history`: Show conversation history
                """))
                continue
                
            elif user_input.lower() == '/clear':
                self.console.print(f"[yellow]{self.clear_history()}[/yellow]")
                continue
                
            elif user_input.lower() == '/history':
                self.console.print("[bold]Conversation History:[/bold]")
                for msg in self.conversation_history:
                    role = msg["role"].capitalize()
                    content = msg["content"]
                    self.console.print(f"[bold]{role}:[/bold] {content}\n")
                continue
                
            elif user_input.lower().startswith('/system '):
                new_prompt = user_input[8:].strip()
                self.console.print(f"[yellow]{self.set_system_prompt(new_prompt)}[/yellow]")
                continue
                
            elif user_input.lower().startswith('/temp '):
                try:
                    temp = float(user_input[6:].strip())
                    if 0.0 <= temp <= 1.0:
                        self.temperature = temp
                        self.console.print(f"[yellow]Temperature set to {temp}[/yellow]")
                    else:
                        self.console.print("[red]Temperature must be between 0.0 and 1.0[/red]")
                except ValueError:
                    self.console.print("[red]Invalid temperature value[/red]")
                continue
                
            elif user_input.lower().startswith('/tokens '):
                try:
                    tokens = int(user_input[8:].strip())
                    if tokens > 0:
                        self.max_tokens = tokens
                        self.console.print(f"[yellow]Max tokens set to {tokens}[/yellow]")
                    else:
                        self.console.print("[red]Max tokens must be positive[/red]")
                except ValueError:
                    self.console.print("[red]Invalid max tokens value[/red]")
                continue
                
            # Process normal input
            self.console.print("[cyan]Generating response...[/cyan]")
            response, elapsed = self.generate_response(user_input)
            
            # Display response
            self.console.print(f"\n[DeepSeek]: {response}")
            self.console.print(f"[dim](Response generated in {elapsed:.2f}s)[/dim]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepSeek Chatbot")
    parser.add_argument("--model", type=str, default="deepseek-ai/deepseek-llm-7b-chat", 
                        help="Model to use for the chatbot")
    parser.add_argument("--temp", type=float, default=0.7, help="Temperature for generation")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p value for generation")
    parser.add_argument("--max_tokens", type=int, default=512, help="Maximum tokens to generate")
    
    args = parser.parse_args()
    
    # Initialize and run chatbot
    chatbot = DeepSeekChatBot(
        model_name=args.model,
        temperature=args.temp,
        top_p=args.top_p, 
        max_tokens=args.max_tokens
    )
    chatbot.run()
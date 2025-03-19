import os
import torch
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import time
import argparse
from rich.console import Console
import json
from werkzeug.utils import secure_filename
import uuid
from MemoryManager  import MemoryManager
memory_manager = MemoryManager()

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max upload size

class DeepSeekLLM:
    def __init__(self, model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", temperature=0.7, top_p=0.9, max_tokens=512):
        self.console = Console()
        self.console.print("[bold blue]Initializing DeepSeek LLM API Backend...[/bold blue]")
        
        # Initialize common attributes regardless of model loading success
        # Set generation parameters
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        
        # Initialize conversation histories
        self.conversations = {}
        self.system_prompt = "You are a helpful AI assistant that provides accurate and thoughtful answers."
        
        # Check available GPU memory
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Convert to GB
            self.console.print(f"[yellow]Available GPU memory: {gpu_memory:.2f} GB[/yellow]")
        else:
            gpu_memory = 0
            self.console.print("[yellow]No GPU detected, using CPU only[/yellow]")
       
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        
        self.console.print(f"[yellow]Loading model: {model_name}...[/yellow]")
        
        try:
            # Simplified loading approach to avoid conflicts
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.float16
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.console.print("[bold green]API backend initialized successfully![/bold green]")
        except Exception as e:
            self.console.print(f"[bold red]Error loading model: {str(e)}[/bold red]")
            
            # Try with even simpler configuration
            self.console.print("[yellow]Attempting with alternative configuration...[/yellow]")
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    load_in_4bit=True,
                    device_map="auto",
                    torch_dtype=torch.bfloat16
                )
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.console.print("[green]Successfully loaded with alternative config[/green]")
            except Exception as e2:
                # Try a smaller model as last resort
                self.console.print(f"[red]Second attempt failed: {str(e2)}[/red]")
                self.console.print("[yellow]Attempting with smaller model...[/yellow]")
                try:
                    fallback_model = "deepseek-ai/deepseek-coder-1.3b-instruct"
                    self.model = AutoModelForCausalLM.from_pretrained(
                        fallback_model,
                        device_map="auto",
                        torch_dtype=torch.float16
                    )
                    self.tokenizer = AutoTokenizer.from_pretrained(fallback_model)
                    self.console.print(f"[green]Loaded fallback model: {fallback_model}[/green]")
                except Exception as e3:
                    self.console.print(f"[bold red]All attempts failed. Last error: {str(e3)}[/bold red]")
                    raise RuntimeError("Failed to load model after multiple attempts")
    
    def format_prompt(self, conversation_history):
        """Format conversation history into a proper prompt for DeepSeek model"""
        messages = [{"role": "system", "content": self.system_prompt}]
        
        for message in conversation_history:
            messages.append(message)
            
        # DeepSeek models use a specific chat template
        prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        return prompt
    
    def generate_response(self, user_input, session_id=None):
        # Create or get conversation history for this session
        if session_id is None:
            session_id = "default"
        
        if session_id not in self.conversations:
            self.conversations[session_id] = []
        
        # Add user message to history
        self.conversations[session_id].append({"role": "user", "content": user_input})
        
        # Format full conversation with history
        prompt = self.format_prompt(self.conversations[session_id])
        
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
        self.conversations[session_id].append({"role": "assistant", "content": response_text})
        
        # Calculate response time
        elapsed_time = time.time() - start_time
        
        return {
            "response": response_text,
            "elapsed_time": elapsed_time,
            "input_tokens": inputs.input_ids.shape[1],
            "output_tokens": len(new_tokens)
        }
    
    def clear_history(self, session_id=None):
        """Clear conversation history for a specific session"""
        if session_id is None:
            session_id = "default"
        
        if session_id in self.conversations:
            self.conversations[session_id] = []
        
        return {"status": "success", "message": f"Conversation history cleared for session {session_id}"}
    
    def set_system_prompt(self, new_prompt):
        """Update the system prompt"""
        self.system_prompt = new_prompt
        return {"status": "success", "message": f"System prompt updated to: {new_prompt}"}

# Initialize the model
llm = DeepSeekLLM()

@app.route('/api/generate', methods=['POST'])
def generate():
    """API endpoint for text generation"""
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    data = request.get_json()
    
    # Check required parameters
    if 'text' not in data:
        return jsonify({"error": "Missing 'text' parameter"}), 400
    
    # Get optional parameters
    session_id = data.get('session_id', None)
    temperature = data.get('temperature', None)
    top_p = data.get('top_p', None)
    max_tokens = data.get('max_tokens', None)
    
    # Temporarily update generation parameters if provided
    temp_temp, temp_top_p, temp_max_tokens = llm.temperature, llm.top_p, llm.max_tokens
    
    if temperature is not None:
        try:
            temp = float(temperature)
            if 0.0 <= temp <= 1.0:
                llm.temperature = temp
        except (ValueError, TypeError):
            pass
            
    if top_p is not None:
        try:
            top_p_val = float(top_p)
            if 0.0 <= top_p_val <= 1.0:
                llm.top_p = top_p_val
        except (ValueError, TypeError):
            pass
            
    if max_tokens is not None:
        try:
            tokens = int(max_tokens)
            if tokens > 0:
                llm.max_tokens = tokens
        except (ValueError, TypeError):
            pass
    
    try:
        # Generate response
        result = llm.generate_response(data['text'], session_id)
        
        # Restore original parameters
        llm.temperature, llm.top_p, llm.max_tokens = temp_temp, temp_top_p, temp_max_tokens
        
        return jsonify(result)
    
    except Exception as e:
        # Restore original parameters
        llm.temperature, llm.top_p, llm.max_tokens = temp_temp, temp_top_p, temp_max_tokens
        
        return jsonify({"error": str(e)}), 500

@app.route('/api/clear', methods=['POST'])
def clear_history():
    """API endpoint to clear conversation history"""
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    data = request.get_json()
    session_id = data.get('session_id', None)
    
    result = llm.clear_history(session_id)
    return jsonify(result)

@app.route('/api/system_prompt', methods=['POST'])
def set_system_prompt():
    """API endpoint to set system prompt"""
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    data = request.get_json()
    
    if 'prompt' not in data:
        return jsonify({"error": "Missing 'prompt' parameter"}), 400
    
    result = llm.set_system_prompt(data['prompt'])
    return jsonify(result)

@app.route('/api/process_paper_cpu', methods=['POST'])
def process_paper_cpu():
    """CPU-only endpoint for processing very large papers"""
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    data = request.get_json()
    
    # Required fields check
    if 'title' not in data or 'content' not in data:
        return jsonify({"error": "Missing required fields: 'title' and 'content' are required"}), 400
    
    # Force CPU processing
    original_cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    try:
        # Process content in smaller chunks if necessary
        title = data.get('title')
        content = data.get('content', '')
        abstract = data.get('abstract', '')
        
        # For very large content, process in chunks
        chunk_size = 15000  # characters
        if len(content) > chunk_size:
            # Split content into chunks
            chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
            
            # Process each chunk separately
            results = []
            for i, chunk in enumerate(chunks):
                chunk_prompt = f"This is part {i+1} of {len(chunks)} of the document.\n\nTitle: {title}\n\nAbstract: {abstract}\n\nContent: {chunk}"
                
                # Generate summary for this chunk
                session_id = data.get('session_id', f"cpu-paper-{uuid.uuid4()}")
                chunk_result = llm.generate_response(chunk_prompt, f"{session_id}-chunk-{i}")
                results.append(chunk_result['response'])
                
                # Clean up between chunks
                gc.collect()
            
            # Combine results
            combined_content = "\n\n".join(results)
            
            # Final summary of the combined chunks
            final_prompt = f"Based on these document parts, provide a final summary:\n\n{combined_content[:10000]}"
            final_result = llm.generate_response(final_prompt, session_id)
            
            # Add metadata
            final_result['paper_title'] = title
            final_result['output_format'] = 'chunked_summary'
            final_result['chunks_processed'] = len(chunks)
            
            return jsonify(final_result)
        else:
            # For smaller content, process directly (similar to regular process_paper)
            # Copy relevant logic from process_paper function
            pass
            
    finally:
        # Restore original CUDA settings
        os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda_visible_devices

# Update your process_paper function with the decorator and content chunking
@app.route('/api/process_paper', methods=['POST'])
# @manage_memory  # Commented out as it is not defined
def process_paper():
    """API endpoint for processing research papers and technical documents"""
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    data = request.get_json()
    
    # Required fields
    if 'title' not in data or 'content' not in data:
        return jsonify({"error": "Missing required fields: 'title' and 'content' are required"}), 400
    
    # Check memory before proceeding
    percent_used, allocated, total = memory_manager.check_gpu_memory()
    
    # If memory is above 80%, return busy status
    if percent_used > 0.8:
        return jsonify({
            "error": "Server is currently processing other requests. Please try again later.",
            "memory_info": {
                "percent_used": round(percent_used * 100, 2),
                "allocated_gb": round(allocated, 2),
                "total_gb": round(total, 2)
            }
        }), 503  # Service Unavailable
    
    # Get paper data
    title = data.get('title')
    abstract = data.get('abstract', '')
    content = data.get('content', '')
    sections = data.get('sections', {})
    figures = data.get('figures', [])
    tables = data.get('tables', [])
    references = data.get('references', [])
    
    # Check content length - enforce limits to prevent OOM errors
    max_safe_chars = 20000  # Safe limit for content length
    if len(content) > max_safe_chars:
        # Truncate content
        content = content[:max_safe_chars] + "\n\n[Content truncated due to length constraints]"
    
    # Optional processing parameters
    output_format = data.get('output_format', 'complete')  # complete, summary, highlights
    session_id = data.get('session_id', f"paper-{uuid.uuid4()}")
    max_tokens = int(data.get('max_tokens', 512))  # Reduce default output tokens
    temperature = float(data.get('temperature', 0.4))  # Lower temperature for more factual responses
    
    # Format the prompt based on the requested output
    if 'prompt' in data:
        system_prompt = "You are a scientific research assistant that extracts key insights from technical papers and presents them as bullet points."
        prompt = data['prompt']
    else:
        # Format the prompt based on the requested output
        if output_format == 'summary':
            system_prompt = "You are a scientific research assistant specializing in summarizing technical papers. Provide concise, accurate summaries that capture the key points."
            prompt = f"""Please summarize the following research paper:
            
Title: {title}

Abstract: {abstract}

Content: {content}

Your summary should include:
1. Main research question or objective
2. Methodology overview
3. Key findings and results
4. Main conclusions and implications
"""
        elif output_format == 'highlights':
            system_prompt = "You are a scientific research assistant that extracts key insights from technical papers and presents them as bullet points."
            prompt = f"""Extract the key highlights from this research paper:
            
Title: {title}

Abstract: {abstract}

Content: {content}

Please provide:
1. 5 key takeaways as bullet points
2. The main innovation or contribution
3. Practical applications
4. Limitations mentioned in the paper
5. Suggested future research directions
"""

        else:  # complete analysis
            system_prompt = "You are a scientific research assistant that provides thorough analysis of technical papers with academic rigor and precision."
            prompt = f"""Analyze the following research paper in detail:
            
Title: {title}

Abstract: {abstract}

Content: {content}

Sections: {json.dumps(sections) if sections else "Not provided"}

Please provide:
1. Comprehensive summary
2. Evaluation of methodology
3. Analysis of results and their significance
4. Critical assessment of strengths and limitations
5. Connections to related work in the field
6. Implications for future research
"""
        
        # Add figures and tables if provided
        if figures or tables:
            prompt += "\n\nAdditional visual elements from the paper:"
            
            if figures:
                prompt += "\n\nFIGURES:\n" + "\n".join([f"- Figure {i+1}: {fig.get('caption', 'No caption')}" 
                                                    for i, fig in enumerate(figures)])
            
            if tables:
                prompt += "\n\nTABLES:\n" + "\n".join([f"- Table {i+1}: {table.get('caption', 'No caption')}" 
                                                    for i, table in enumerate(tables)])
    
    # Add references if provided
    if references:
        prompt += "\n\nREFERENCES:\n" + "\n".join([f"[{i+1}] {ref}" for i, ref in enumerate(references)])
    
    # Temporarily set system prompt for this request
    original_system_prompt = llm.system_prompt
    llm.system_prompt = system_prompt
    
    try:
        # Generate response
        result = llm.generate_response(prompt, session_id)
        
        # Add metadata to result
        result["paper_title"] = title
        result["output_format"] = output_format
        
        # Restore original system prompt
        llm.system_prompt = original_system_prompt
        
        return jsonify(result)
    
    except Exception as e:
        # Restore original system prompt
        llm.system_prompt = original_system_prompt
        return jsonify({"error": str(e)}), 500


@app.route('/api/batch_process', methods=['POST'])
def batch_process():
    """API endpoint for processing multiple papers in batch"""
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    data = request.get_json()
    
    if 'papers' not in data or not isinstance(data['papers'], list):
        return jsonify({"error": "Request must contain 'papers' array"}), 400
    
    # Common parameters
    output_format = data.get('output_format', 'summary')
    max_tokens = int(data.get('max_tokens', 1024))
    temperature = float(data.get('temperature', 0.4))
    
    # Create a batch session ID
    batch_session_id = f"batch-{uuid.uuid4()}"
    
    # Process each paper
    results = []
    for i, paper in enumerate(data['papers']):
        if not isinstance(paper, dict) or 'title' not in paper or 'content' not in paper:
            results.append({
                "index": i,
                "error": "Invalid paper format. Each paper must have 'title' and 'content'"
            })
            continue
        
        # Create a paper-specific session ID
        paper_session_id = f"{batch_session_id}-paper-{i}"
        
        # Add batch parameters if not specified in the individual paper
        if 'output_format' not in paper:
            paper['output_format'] = output_format
        if 'max_tokens' not in paper:
            paper['max_tokens'] = max_tokens
        if 'temperature' not in paper:
            paper['temperature'] = temperature
        
        # Add session ID
        paper['session_id'] = paper_session_id
        
        try:
            # Call the process_paper function directly with the paper data
            with app.test_request_context(
                '/api/process_paper', 
                method='POST',
                json=paper
            ):
                response = process_paper()
                if hasattr(response, 'json'):
                    result = response.json
                else:
                    result = json.loads(response.get_data(as_text=True))
                
                # Add index and title for reference
                result['index'] = i
                result['title'] = paper['title']
                results.append(result)
                
        except Exception as e:
            results.append({
                "index": i,
                "title": paper['title'],
                "error": str(e)
            })
    
    return jsonify({
        "batch_id": batch_session_id,
        "total_papers": len(data['papers']),
        "processed_papers": len([r for r in results if 'error' not in r]),
        "failed_papers": len([r for r in results if 'error' in r]),
        "results": results
    })

@app.route('/health', methods=['GET'])
def health_check():
    """API endpoint for health check"""
    # Check GPU memory
    if hasattr(memory_manager, 'check_gpu_memory'):
        percent_used, allocated, total = memory_manager.check_gpu_memory()
    else:
        percent_used, allocated, total = 0, 0, 0
    
    # Get model info safely
    try:
        model_name = llm.model.config.name_or_path
    except:
        model_name = "Unknown"
    
    # Get system memory info
    system_memory = psutil.virtual_memory()
    
    return jsonify({
        "status": "healthy", 
        "model": model_name,
        "memory": {
            "gpu_percent_used": round(percent_used * 100, 2) if torch.cuda.is_available() else 0,
            "gpu_allocated_gb": round(allocated, 2) if torch.cuda.is_available() else 0,
            "gpu_total_gb": round(total, 2) if torch.cuda.is_available() else 0,
            "system_percent_used": system_memory.percent,
            "system_available_gb": round(system_memory.available / (1024**3), 2)
        }
    })

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DeepSeek LLM API")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", 
                        help="Model to use for the API")
    parser.add_argument("--temp", type=float, default=0.7, help="Default temperature for generation")
    parser.add_argument("--top_p", type=float, default=0.9, help="Default top-p value for generation")
    parser.add_argument("--max_tokens", type=int, default=512, help="Default maximum tokens to generate")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind the server to")
    parser.add_argument("--debug", action="store_true", help="Run Flask in debug mode")
    parser.add_argument("--cpu-only", action="store_true", 
                        help="Run in CPU-only mode for better stability with larger documents")
    parser.add_argument("--low-memory", action="store_true", 
                        help="Run with aggressive memory optimization for systems with limited GPU RAM")
    
    args = parser.parse_args()
    
    # Force CPU mode if requested
    if args.cpu_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    # Initialize memory manager
    memory_manager = MemoryManager()
    
    # Initialize LLM with command line arguments, use smaller model in low-memory mode
    if args.low_memory:
        model_to_load = "deepseek-ai/deepseek-coder-1.3b-instruct"
    else:
        model_to_load = args.model
    
    llm = DeepSeekLLM(
        model_name=model_to_load,
        temperature=args.temp,
        top_p=args.top_p,
        max_tokens=args.max_tokens
    )
    
    # Run the Flask app with memory cleanup on startup
    memory_manager.cleanup_memory()
    app.run(host=args.host, port=args.port, debug=args.debug)
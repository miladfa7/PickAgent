import gradio as gr
import os
import tempfile
from PIL import Image
import numpy as np
import time
from pathlib import Path

# Import the PickAgent and utilities
from openvla import OpenVLA
from PIL import Image
import numpy as np
import os
import mediapy
import sapien.core as sapien
from mbodied.robots import Robot
from robot_utils import (
    get_libero_env,
    GenerateConfig,
    get_image_resize_size,
    get_task,
    get_libero_dummy_action,
    get_libero_image,
    normalize_gripper_action,
    invert_gripper_action,
    save_rollout_video,
    quat2axisangle
)

class PickAgent:
    def __init__(self, task="libero_object", task_id=0, image_resize=1024, output_video=None):
        """
        Initialize the OpenVLA agent with configuration parameters.
        """
        self.cfg = GenerateConfig()
        self.cfg.task_suite_name = task
        self.cfg.num_steps_wait = 20
        self.output_video = output_video or "outputs/videos"
        self.task_name = task
        self.task_id = task_id
        self.custom_resize = image_resize
        self.cfg.unnorm_key = self.cfg.task_suite_name

        # Initialize OpenVLA
        model_name = self.get_model_name(task)
        self.openvla = OpenVLA(model_name=model_name)
        
        # Load task and environment
        self.setup_environment()
    
    def get_model_name(self, task):
        model_name = "openvla/openvla-7b"
        if task=="libero_object":
            model_name = "openvla/openvla-7b-finetuned-libero-object"
        elif task=="libero_spatial":
            model_name = "openvla/openvla-7b-finetuned-libero-spatial"
        elif task=="libero_goal":
            model_name = "openvla/openvla-7b-finetuned-libero-goal"
        elif task=="libero_10":
            model_name = "openvla/openvla-7b-finetuned-libero-10"
        elif task=="general":
            model_name = "openvla/openvla-7b"
        return model_name

    def setup_environment(self):
        """Set up the task and environment."""

        if self.task_name == "libero_spatial":
            self.max_steps = 220
        elif self.task_name == "libero_object":
            self.max_steps = 280
        elif self.task_name == "libero_goal":
            self.max_steps = 300
        elif self.task_name == "libero_10":
            self.max_steps = 520
        elif self.task_name == "libero_90":
            self.max_steps = 400
        else:
            self.max_steps = 280

        self.task, self.task_suite = get_task(self.cfg, self.task_id)
        self.initial_states = self.task_suite.get_task_init_states(self.task_id)
        self.env, self.task_description = get_libero_env(self.task, self.cfg.model_family, resolution=256)

        if self.custom_resize:
            self.resize_size = self.custom_resize
        else:
            self.resize_size = get_image_resize_size(self.cfg)
        
        # Set unnormalization key
        if hasattr(self.openvla, 'model') and hasattr(self.openvla.model, 'norm_stats'):
            if self.cfg.unnorm_key not in self.openvla.model.norm_stats and f"{self.cfg.unnorm_key}_no_noops" in self.openvla.model.norm_stats:
                self.cfg.unnorm_key = f"{self.cfg.unnorm_key}_no_noops"
    
    def get_first_frame(self):
        self.env.reset()
        obs = self.env.set_init_state(self.initial_states[0])
        
        for _ in range(self.cfg.num_steps_wait):
            obs, _, _, _ = self.env.step(get_libero_dummy_action(self.cfg.model_family))
        
        _, img = self.process_observation(obs)
        return img, self.task_description
    
    def process_observation(self, obs):
        """
        Process raw observation into model input format.
        """
        # Get preprocessed image
        img = get_libero_image(obs, self.resize_size)
        
        # Prepare observations dict with state information
        observation = {
            "full_image": img,
            "state": np.concatenate(
                (
                    obs["robot0_eef_pos"], 
                    quat2axisangle(obs["robot0_eef_quat"]), 
                    obs["robot0_gripper_qpos"]
                )
            ),
        }
        
        return observation, img
    
    def run_simulation(self, prompt, episode_idx=0):
        """
        Run a full simulation episode with the given prompt.
        """
        # Reset environment and set initial state
        self.env.reset()
        obs = self.env.set_init_state(self.initial_states[episode_idx])
        
        t = 0
        replay_images = []
        success = False
        
        print(f"Starting simulation with prompt: '{prompt}'")
        
        while t < self.max_steps + self.cfg.num_steps_wait:
            # Wait period with dummy actions
            if t < self.cfg.num_steps_wait:
                obs, reward, done, info = self.env.step(get_libero_dummy_action(self.cfg.model_family))
                t += 1
                continue
            
            # Process observation
            observation, img = self.process_observation(obs)
            
            replay_images.append(img)
            
            # Get action from OpenVLA model
            action = self.openvla.get_action(
                observation=observation,
                prompt=prompt,
                unnorm_key=self.cfg.unnorm_key
            )
            
            # Log action for debugging
            print(f"Action = {action}")
            
            # Normalize and apply action
            action = normalize_gripper_action(action, binarize=True)
            action = invert_gripper_action(action)
            obs, reward, done, info = self.env.step(action.tolist())
            
            t += 1
            
            # Check for episode completion
            if done:
                success = True
                print(f"Episode completed successfully in {t} steps!")
                break
                
        if t >= self.max_steps + self.cfg.num_steps_wait:
            print("Episode reached maximum steps without completion.")
        
        # Save replay video
        video_path = save_rollout_video(
            replay_images, 
            self.task_id, 
            success=success, 
            task_description=prompt,
            output_dir=self.output_video
        )
        print(f"Saved replay video to {video_path}")
        
        return video_path, success


agent_instances = {}

def get_model_info(task):
    if task == "libero_object":
        return "Will load: openvla/openvla-7b-finetuned-libero-object"
    elif task == "libero_spatial":
        return "Will load: openvla/openvla-7b-finetuned-libero-spatial"
    elif task == "libero_goal":
        return "Will load: openvla/openvla-7b-finetuned-libero-goal"
    elif task == "libero_10":
        return "Will load: openvla/openvla-7b-finetuned-libero-10"
    elif task == "general":
        return "Will load: openvla/openvla-7b"
    return ""


def initialize_agent(task, task_id):
    """Initialize agent"""

    image_resize = 512
    output_path = "outputs/videos"
    
    key = f"{task}_{task_id}_{image_resize}"
    if key not in agent_instances:   
        os.makedirs(output_path, exist_ok=True)
        
        # Initialize the agent
        agent_instances[key] = PickAgent(
            task=task,
            task_id=task_id,
            image_resize=image_resize,
            output_video=output_path
        )
    
    return agent_instances[key]


def update_preview_and_prompt(task, task_id):
    agent.task_id = task_id
    agent.setup_environment()
    img, desc = agent.get_first_frame()
    model_info = get_model_info(task)
    return img, model_info, desc

def run_simulation(prompt):
    video_path, success = agent.run_simulation(prompt)
    status = "✅ Success!" if success else "❌ Failed!"
    return video_path, status

with gr.Blocks(title="PickAgent") as demo:
    gr.Markdown("# PickAgent: OpenVLA-powered Pick and Place Agent (Simulation)")
    gr.Markdown("Control robotic agents using natural language prompt with OpenVLA-powered inference")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Configuration")
            
            task = gr.Dropdown(
                choices=list(task_options.keys()), 
                label="Task", 
                value="libero_object",
                info="Select the task name"
            )
            model_info = gr.Markdown(f"Will load: openvla/openvla-7b-finetuned-libero-object")
            
            task_id = gr.Number(
                label="Task ID", 
                precision=0,  
                value=0,
                minimum=0,    
                maximum=9, 
                info="Select the specific task instance"
            )
                    
            prompt = gr.Textbox(
                label="Prompt", 
                placeholder="Enter your instructions for the robot",
                info="Natural language instruction for the robot"
            )
            
            preview_btn = gr.Button("Preview Environment")
            run_btn = gr.Button("Run Simulation", variant="primary")
            
        # Right side - Outputs with tabs
        with gr.Column(scale=1):
            gr.Markdown("### Environment & Results")
            
            with gr.Tabs():
                with gr.TabItem("Preview"):
                    preview_image = gr.Image(label="Environment Preview")
                
                with gr.TabItem("Simulation"):
                    status = gr.Markdown("Status: Ready")
                    video_output = gr.Video(label="Simulation Result", width=512, height=512)
    
    agent = initialize_agent(task.value, task_id.value)

    preview_btn.click(
        fn=update_preview_and_prompt,
        inputs=[task, task_id],
        outputs=[preview_image, model_info, prompt]
    )
    
    run_btn.click(
        fn=run_simulation,
        inputs=[prompt],
        outputs=[video_output, status]
    )

if __name__ == "__main__":
    demo.launch()

<img width="1920" height="1080" alt="image (3)" src="https://github.com/user-attachments/assets/1588a09d-d87d-4066-81cb-e2643e0ed201" />
"Real-time Obstacle Avoidance & Path Planning" Visualized global/local costmaps in RViz. The robot successfully navigates dynamic environments, demonstrating the capability to safely transport chemical samples without collisions in a crowded lab."

<img width="1920" height="1080" alt="image (4)" src="https://github.com/user-attachments/assets/1e4a4b31-7987-4066-baef-872d9cbd13d6" />
## ⚙️ Navigation Parameter Tuning
> *Fine-tuned ROS 2 Nav2 parameters (YAML) to optimize localization accuracy and obstacle avoidance behavior.*

**Key Optimizations:**
* **AMCL Localization:** Adjusted motion model noise parameters (`alpha1` ~ `alpha5`) to reduce pose estimation errors during rotational movements.
* **Costmap Configuration:** Tuned `inflation_radius` and `cost_scaling_factor` to ensure the robot maintains a safe distance from delicate lab equipment while navigating narrow passages.
* **Beam Skip:** Enabled `beam_skip` to filter out dynamic obstacles (e.g., walking humans) from the localization process, ensuring stability in busy environments.

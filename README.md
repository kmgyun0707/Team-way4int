<img width="1920" height="1080" alt="image (3)" src="https://github.com/user-attachments/assets/1588a09d-d87d-4066-81cb-e2643e0ed201" />
"Real-time Obstacle Avoidance & Path Planning" Visualized global/local costmaps in RViz. The robot successfully navigates dynamic environments, demonstrating the capability to safely transport chemical samples without collisions in a crowded lab."

<img width="1920" height="1080" alt="image (5)" src="https://github.com/user-attachments/assets/4ce0e257-04b7-4373-8d1d-5d017a75a015" />
## ⚙️ Navigation Parameter Tuning      

> *Fine-tuned ROS 2 Nav2 parameters (YAML) to optimize localization accuracy and obstacle avoidance behavior.*

**Key Optimizations:**
* **AMCL Localization:** Adjusted motion model noise parameters (`alpha1` ~ `alpha5`) to reduce pose estimation errors during rotational movements.
* **Costmap Configuration:** Tuned `inflation_radius` and `cost_scaling_factor` to ensure the robot maintains a safe distance from delicate lab equipment while navigating narrow passages.
* **Beam Skip:** Enabled `beam_skip` to filter out dynamic obstacles (e.g., walking humans) from the localization process, ensuring stability in busy environments.

<img width="2560" height="1600" alt="image (6)" src="https://github.com/user-attachments/assets/c23c72f1-8af3-4a6a-b5f9-41bb23f61540" />
## 🛡️ Robust Localization & Autonomous Recovery
> *Analysis of AMCL particle convergence (red dots) and autonomous recovery behaviors during long-term operation.*

To ensure reliable 24/7 operation in an **Autonomous Laboratory**, the robot must maintain precise localization and handle unexpected navigation failures without human intervention.
* **Precise Localization (AMCL):** The dense cluster of red particles around the robot confirms high-confidence localization. This precision is critical for **robot arm docking** tasks when transferring chemical samples between stations.
* **Scan Matching Verification:** The alignment between real-time LiDAR scans (red lines) and the static costmap (purple) validates the **Sim-to-Real sensor calibration**, ensuring the robot accurately perceives lab benches and equipment.
* **Fault Tolerance (Recoveries):** As shown in the panel (`Recoveries: 9`), the system successfully triggered autonomous recovery behaviors (e.g., clearing costmaps, backing up) to resolve path planning failures, guaranteeing continuous workflow execution in dynamic environments.

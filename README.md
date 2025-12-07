# Guided-Trajectory-Neural-Net
This project builds a machine-learning surrogate model using data generated from a quasi-6-DoF physics simulator replacing numerical integration with a neural network that can predict full 3-D projectile trajectories—including altitude, downrange motion, and lateral deflection hundreds of times faster while maintaining meter-level accuracy.

============================================================
FINAL METRICS
============================================================
  mean_trajectory_error: 0.2518 m
  max_trajectory_error: 2.9245 m
  mean_final_position_error: 0.4159 m
  std_final_position_error: 0.2601 m
  std_final_position_error: 0.2601 m
  median_final_position_error: 0.3590 m
  95th_percentile_error: 0.9205 m


  Neural network: 12,923 trajectories/sec
  Neural network: 12,923 trajectories/sec
  Physics sim: ~30 trajectories/sec
  Physics sim: ~30 trajectories/sec
  Speedup: 431x faster!
  Speedup: 431x faster!
  

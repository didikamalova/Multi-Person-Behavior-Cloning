# Behavior Cloning from Multiple Demonstrators

Traditionally, behavior cloning has focused on learning a particular skill in isolated demonstrations from a single demonstration. To learn more skills, robots would need (a) large number of robot-tailored expert demonstrations that are provided (b) from multiple agents. However, policies learned through a single demonstration have shown to deteriorate in performance when evaluated with multiple demonstrators. This happens because people’s actions tend to be, multimodal, non-deterministic and continuous.

By conditioning the BC agent on who’s collecting the data, we can work with data under the same distribution. We recover conditional distribution which no longer violates the unimodality requirement.

# Didactic Experiments
The experiment setup involves the robotic arm and a gripper placed on a table with multiple objects. The task of an agent is to grab an object and place it on the tray.

• 5 different demonstrations decreasing in performance (increasing noise) were simulated

• 11 policies, involving 10 conditioned & unconditioned policies each trained on the single demonstrator data and 1 unconditioned policy trained on the combined data, were trained


It was found that individually recovered policies don’t necessarily improve in performance. This might happen due to the data overlap between the demonstrators since neural networks are not fully expressive and/or weighting is needed.

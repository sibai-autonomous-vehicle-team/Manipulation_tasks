import gymnasium as gym
import mani_skill.envs

def find_exact_bowl_and_types():
    env = gym.make("UnitreeG1PlaceAppleInBowl-v1")
    obs, _ = env.reset()
    
    # Find bowl - check these in order:
    print("BOWL LOCATION:")
    if hasattr(env, 'bowl'): print(f"✓ env.bowl = {env.bowl}")
    if hasattr(env, '_bowl'): print(f"✓ env._bowl = {env._bowl}")
    if hasattr(env, 'target_bowl'): print(f"✓ env.target_bowl = {env.target_bowl}")
    
    # Find exact bowl actor name
    all_actors = env.scene.get_all_actors()
    print(f"\nBOWL ACTORS: {[a.name for a in all_actors if 'bowl' in a.name.lower()]}")
    
    # Find hand link names
    robot = env.agent.robot
    hand_links = [link.name for link in robot.get_links() 
                  if any(word in link.name.lower() for word in ['hand', 'finger', 'gripper', 'palm'])]
    print(f"\nHAND LINKS: {hand_links}")
    
    # Test contact force type
    links = robot.get_links()
    if len(links) >= 2:
        test_forces = env.scene.get_pairwise_contact_forces(links[0], links[1])
        print(f"\nCONTACT FORCE TYPE: {type(test_forces)}")
    
    env.close()

find_exact_bowl_and_types()
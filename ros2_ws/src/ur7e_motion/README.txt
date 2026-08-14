流程:
HOME -> /hik_camera/run_once -> 等待新目标 -> OPEN -> PREGRASP
-> GRASP -> CLOSE -> LIFT -> HOME

新增:
 /ur7e/move_to_target
 /ur7e/set_home_here
 /ur7e/home_status
 /ur7e/go_home

调试模式:
 use_separate_plan_execute=true

正常模式:
 use_separate_plan_execute=false

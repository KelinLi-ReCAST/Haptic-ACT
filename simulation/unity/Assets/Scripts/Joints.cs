using UnityEngine;
using System.Collections.Generic;
using System.IO;
using System.Diagnostics;
using Mujoco;

public class ActuatorCommandSaver : MonoBehaviour
{
    // Array to store actuator commands, assuming you have 14 actuators
    private float[] actuatorCommands = new float[14];
    private List<float[]> commandDataList = new List<float[]>();
    private int currentFrame = 0;
    private int SaveInterval = 10;
    private string filePath;
    string saveFolderPathJoint = "Assets/Demonstrations/Joints";

    // References to your actuator components
    public GameObject act1;
    public GameObject act2;
    public GameObject act3;
    public GameObject act4;
    public GameObject act5;
    public GameObject act6;
    public GameObject act7;
    public GameObject thumb_yaw;
    public GameObject thumb_pitch;
    public GameObject index;
    public GameObject middle;
    public GameObject ring;
    public GameObject pinky;

    void Start()
    {
        // 设置CSV文件路径，这里假设保存到Assets文件夹下的data.csv文件中
        filePath = Path.Combine(saveFolderPathJoint, "data.csv");

        if (!Directory.Exists(saveFolderPathJoint))
        {
            Directory.CreateDirectory(saveFolderPathJoint);
        }

        // 如果文件已存在，则清空内容，否则创建新文件
        if (File.Exists(filePath))
        {
            File.Delete(filePath);
        }
    }

    void Update()
    {
        currentFrame++;
        // 当达到保存间隔时，保存数据到CSV文件
        if (currentFrame % SaveInterval == 0)
        {
            actuatorCommands[13] = currentFrame;
            SaveDataToCSV();
        }
    }

    void SaveDataToCSV()
    {
        // Update actuator commands with actual values from your actuators

        var mjAct1 = act1.GetComponent<MjActuator>();
        var mjAct2 = act2.GetComponent<MjActuator>();
        var mjAct3 = act3.GetComponent<MjActuator>();
        var mjAct4 = act4.GetComponent<MjActuator>();
        var mjAct5 = act5.GetComponent<MjActuator>();
        var mjAct6 = act6.GetComponent<MjActuator>();
        var mjAct7 = act7.GetComponent<MjActuator>();
        var mjAct8 = thumb_yaw.GetComponent<MjActuator>();
        var mjAct9 = thumb_pitch.GetComponent<MjActuator>();
        var mjAct10 = index.GetComponent<MjActuator>();
        var mjAct11 = middle.GetComponent<MjActuator>();
        var mjAct12 = ring.GetComponent<MjActuator>();
        var mjAct13 = pinky.GetComponent<MjActuator>();
        
        actuatorCommands[0] = mjAct1.Control;
        actuatorCommands[1] = mjAct2.Control;
        actuatorCommands[2] = mjAct3.Control;
        actuatorCommands[3] = mjAct4.Control;
        actuatorCommands[4] = mjAct5.Control;
        actuatorCommands[5] = mjAct6.Control;
        actuatorCommands[6] = mjAct7.Control;
        actuatorCommands[7] = mjAct8.Control;
        actuatorCommands[8] = mjAct9.Control;
        actuatorCommands[9] = mjAct10.Control;
        actuatorCommands[10] = mjAct11.Control;
        actuatorCommands[11] = mjAct12.Control;
        actuatorCommands[12] = mjAct13.Control;

        // 打开或创建CSV文件，并写入数据
        using (StreamWriter sw = new StreamWriter(filePath, true))
        {
            // 写入数据头（如果需要）
            if (currentFrame == SaveInterval) // 第一次保存时写入数据头
            {
                string headers = "act1,act2,act3,act4,act5,act6,act7,thumb_yaw,thumb_pitch,index,middle,ring,pinky,timestep";
                sw.WriteLine(headers);
            }

            // 写入当前数组数据
            string line = string.Join(",", actuatorCommands);
            sw.WriteLine(line);
        }

    }
}

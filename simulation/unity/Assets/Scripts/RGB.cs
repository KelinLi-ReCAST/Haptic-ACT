using UnityEngine;
using System.Collections;
using System.IO;

public class CaptureRGB : MonoBehaviour
{
    string saveFolderPathRGB = "Assets/Demonstrations/RGBImages";

    int saveInterval = 10; // Save every 10 timesteps
    int imageWidth = 480;
    int imageHeight = 480;

    private int currentFrame = 0;


    void Start()
    {
        // Create save folders if they don't exist
        if (!Directory.Exists(saveFolderPathRGB))
        {
            Directory.CreateDirectory(saveFolderPathRGB);
        }
    }

    void Update()
    {
        currentFrame++;

        // Check if it's time to save images
        if (currentFrame % saveInterval == 0)
        {
            SaveRGBImage();
            //SaveDepthImage();
        }
    }

    void SaveRGBImage()
    {
        // Create a RenderTexture to capture the RGB view
        RenderTexture renderTexture = new RenderTexture(imageWidth, imageHeight, 24, RenderTextureFormat.Default);
        GetComponent<Camera>().targetTexture = renderTexture;
        Texture2D texture = new Texture2D(imageWidth, imageHeight, TextureFormat.RGB24, false);

        // Render the RGB view to the texture
        GetComponent<Camera>().Render();

        RenderTexture.active = renderTexture;
        texture.ReadPixels(new Rect(0, 0, imageWidth, imageHeight), 0, 0);
        texture.Apply();

        // Reset the camera's target texture
        GetComponent<Camera>().targetTexture = null;
        RenderTexture.active = null; // Remove active render texture

        // Encode texture to PNG
        byte[] bytes = texture.EncodeToPNG();
        string filePath = Path.Combine(saveFolderPathRGB, "RGBImage_" + Time.frameCount + ".png");
        File.WriteAllBytes(filePath, bytes);

        //Debug.Log("Saved RGB image to " + filePath);

        // Clean up
        Destroy(renderTexture);
        Destroy(texture);
    }
}
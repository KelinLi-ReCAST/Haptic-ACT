using UnityEngine;
using System.IO;

[ExecuteInEditMode]
public class CaptureDepth : MonoBehaviour
{
	[Range(0f, 3f)]
	public float depthLevel = 0.5f;

    string saveFolderPathDepth = "Assets/Demonstrations/DepthImages";

    int saveInterval = 10; // Save every 10 timesteps
    int imageWidth = 480;
    int imageHeight = 480;

    private int currentFrame = 0;

	private Shader _shader;
	private Shader shader
	{
		get { return _shader != null ? _shader : (_shader = Shader.Find("Custom/RenderDepth")); }
	}

	private Material _material;
	private Material material
	{
		get
		{
			if (_material == null)
			{
				_material = new Material(shader);
				_material.hideFlags = HideFlags.HideAndDontSave;
			}
			return _material;
		}
	}

	private void Start ()
	{
        // Create save folders if they don't exist
        if (!Directory.Exists(saveFolderPathDepth))
        {
            Directory.CreateDirectory(saveFolderPathDepth);
        }

		if (!SystemInfo.supportsImageEffects)
		{
			print("System doesn't support image effects");
			enabled = false;
			return;
		}
		if (shader == null || !shader.isSupported)
		{
			enabled = false;
			print("Shader " + shader.name + " is not supported");
			return;
		}

		// turn on depth rendering for the camera so that the shader can access it via _CameraDepthTexture
		GetComponent<Camera>().depthTextureMode = DepthTextureMode.Depth;
	}

    void Update()
    {
        currentFrame++;

        // Check if it's time to save images
        if (currentFrame % saveInterval == 0)
        {
            SaveDepthImage();
        }
    }	

	private void OnDisable()
	{
		if (_material != null)
			DestroyImmediate(_material);
	}
	
	private void OnRenderImage(RenderTexture src, RenderTexture dest)
	{
		if (shader != null)
		{
			material.SetFloat("_DepthLevel", depthLevel);
			Graphics.Blit(src, dest, material);
		}
		else
		{
			Graphics.Blit(src, dest);
		}
	}

    void SaveDepthImage()
    {
        var tempRt = new RenderTexture(imageWidth, imageHeight, 24, RenderTextureFormat.Default);
        
        GetComponent<Camera>().targetTexture = tempRt;
        GetComponent<Camera>().Render();

        RenderTexture.active = tempRt;

        var tex2d = new Texture2D(imageWidth, imageHeight, TextureFormat.ARGB32, false);
        tex2d.ReadPixels(new Rect(0, 0, imageWidth, imageHeight), 0, 0);
        tex2d.Apply();

        var imageData = tex2d.EncodeToPNG();
        string filePath = Path.Combine(saveFolderPathDepth, "DepthImage_" + Time.frameCount + ".png");
        File.WriteAllBytes(filePath, imageData);

        RenderTexture.active = null;
        GetComponent<Camera>().targetTexture = null;
        DestroyImmediate(tempRt);
    }
}
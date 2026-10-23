#!/usr/bin/env python3
"""Render every demo with all four backends, then check Play-mode scene switching.

Run with Unity open in Edit mode and all scene edits saved. Diagnostic images and
metrics go to scratch/demos/verification. No scene assets are saved or modified.
Run again with --play after entering Play mode to test runtime scene switching.
"""
import json
from pathlib import Path
import time
import sys
from benchmark_unity import Unity

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'scratch/demos/verification'
OUTPUT.mkdir(parents=True, exist_ok=True)
unity = Unity()
original = None
playing = "--play" in sys.argv
try:
    original = unity.evaluate('''
if(!UnityEditor.EditorApplication.isPlaying && UnityEditor.SceneManagement.EditorSceneManager.GetActiveScene().isDirty)
    throw new System.Exception("Save scene edits first");
var r=UnityEngine.Object.FindFirstObjectByType<CudaRenderer>();
return new {path=UnityEngine.SceneManagement.SceneManager.GetActiveScene().path, preview=CudaScenePreview.Enabled,
    playing=UnityEditor.EditorApplication.isPlaying,background=UnityEngine.Application.runInBackground,
    width=r.width,height=r.height,backend=(int)r.backend,input=r.InputEnabled,index=DemoScenes.Current};
''')
    assert original['playing']==playing, "Use --play in Play mode, or omit it in Edit mode"
    unity.evaluate('CudaScenePreview.Enabled=false; UnityEngine.Application.runInBackground=true; return true;')
    code = r'''
CudaScenePreview.Enabled=false;
var rows=new System.Collections.Generic.List<object>();
for(int index=0;index<DemoScenes.Paths.Length;index++) {
    UnityEditor.SceneManagement.EditorSceneManager.OpenScene(DemoScenes.Paths[index],UnityEditor.SceneManagement.OpenSceneMode.Single);
    if(UnityEngine.SceneManagement.SceneManager.sceneCount!=1) throw new System.Exception("Extra scene remained loaded");
    var renderer=UnityEngine.Object.FindFirstObjectByType<CudaRenderer>();
    if(renderer==null || renderer.width!=1280 || renderer.height!=720) throw new System.Exception("Missing camera or wrong saved resolution");
    using(var session=new RayTraceSession()) {
        session.UploadScene(96,54);
        byte[] reference=null;
        for(int mode=0;mode<4;mode++) {
            session.ResetAccumulation();
            for(int pass=0;pass<8;pass++) session.Render(renderer.GetComponent<UnityEngine.Camera>(),(CudaRenderer.RenderBackend)mode,renderer.Settings);
            byte[] image=session.Image.GetRawTextureData<byte>().ToArray();
            if(mode==0) reference=image;
            int maximum=0,changed=0; double squared=0,absolute=0; int channels=0;
            for(int p=0;p<image.Length;p++) if(p%4!=3) {
                int d=System.Math.Abs(image[p]-reference[p]);
                maximum=System.Math.Max(maximum,d); absolute+=d; squared+=d*d; channels++;
                if(d>1) changed++;
            }
            if(System.Linq.Enumerable.Sum(System.Linq.Enumerable.Where(image,(value,p)=>p%4!=3),value=>(int)value)==0 || session.SampleCount!=8*renderer.raysPerPixel) throw new System.Exception("Render or accumulation failed");
            // Independent intersections can produce rare divergent stochastic paths.
            // Report those errors rather than requiring identical PNG hashes.
            if(absolute/channels>.5 || (double)changed/channels>.01) throw new System.Exception("Backend images disagree: "+DemoScenes.Names[index]+" / "+mode);
            rows.Add(new {scene=DemoScenes.Names[index],backend=mode,triangles=session.TriangleCount,samples=session.SampleCount,
                loadedScenes=UnityEngine.SceneManagement.SceneManager.sceneCount,maxError=maximum,mae=absolute/channels,rmse=System.Math.Sqrt(squared/channels),changedFraction=(double)changed/channels});
            System.IO.File.WriteAllBytes(System.IO.Path.Combine(__OUTPUT__,index+"-"+mode+".png"),session.Image.EncodeToPNG());
            session.Render(renderer.GetComponent<UnityEngine.Camera>(),(CudaRenderer.RenderBackend)mode,renderer.Settings,true);
            if(session.DenoiseError!=null || session.DenoiseMilliseconds<=0 || session.DisplayedSamples!=session.SampleCount)
                throw new System.Exception("AI stage failed: "+DemoScenes.Names[index]+" / "+mode+" / "+session.DenoiseError);
            System.IO.File.WriteAllBytes(System.IO.Path.Combine(__OUTPUT__,index+"-"+mode+"-ai.png"),session.Image.EncodeToPNG());
        }
        // The display cache must keep accumulating raw samples and must be
        // discarded as soon as the camera changes, even within the same call.
        var camera=renderer.GetComponent<UnityEngine.Camera>();
        session.ResetAccumulation();
        session.Render(camera,CudaRenderer.RenderBackend.Cuda,renderer.Settings,true);
        uint displayed=session.DisplayedSamples;
        session.Render(camera,CudaRenderer.RenderBackend.Cuda,renderer.Settings,true);
        if(session.SampleCount!=2*renderer.raysPerPixel || session.DisplayedSamples<displayed || session.DisplayedSamples>session.SampleCount)
            throw new System.Exception("AI cache lost raw samples");
        uint samples=session.SampleCount;
        session.Render(camera,CudaRenderer.RenderBackend.Cuda,renderer.Settings,false);
        if(session.SampleCount!=samples+renderer.raysPerPixel || session.DisplayedSamples!=session.SampleCount)
            throw new System.Exception("Disabling AI reset the image or retained the filtered texture");
        session.Render(camera,CudaRenderer.RenderBackend.Cuda,renderer.Settings,true);
        if(session.DisplayedSamples!=session.SampleCount) throw new System.Exception("Enabling AI reused an old filtered image");
        var position=camera.transform.position;
        try {
            camera.transform.position+=UnityEngine.Vector3.right*.01f;
            session.Render(camera,CudaRenderer.RenderBackend.Cuda,renderer.Settings,true);
            if(session.SampleCount!=renderer.raysPerPixel || session.DisplayedSamples!=session.SampleCount)
                throw new System.Exception("Camera motion retained old AI history");
        }
        finally { camera.transform.position=position; }
        if(session.DenoiseError!=null) throw new System.Exception(session.DenoiseError);
    }
    UnityEditor.EditorUtility.UnloadUnusedAssetsImmediate();
}
return rows;
'''.replace('__OUTPUT__', json.dumps(str(OUTPUT)))
    # The Pipeline evaluator does not implicitly import LINQ extension methods.
    code = code.replace('session.Image.GetRawTextureData<byte>().ToArray()',
                        'System.Linq.Enumerable.ToArray(session.Image.GetRawTextureData<byte>())')
    if not playing:
        metrics = unity.evaluate(code)
        (OUTPUT / 'images.json').write_text(json.dumps(metrics, indent=2)+'\n')
        print('PASS 20 raw and 20 AI demo/backend renders; toggle, camera reset, sample counts; one scene loaded; saved resolution 1280x720', flush=True)
    def wait_for(expression):
        deadline=time.monotonic()+45
        while time.monotonic()<deadline:
            if unity.evaluate('return '+expression+';'): return
            time.sleep(.3)
        raise TimeoutError(expression)
    switches=[]
    if playing:
        # Start at zero even if the caller originally had a different room open.
        for index in (0,1,2,3,4,0):
            previous=unity.evaluate('var r=UnityEngine.Object.FindFirstObjectByType<CudaRenderer>(); return new {index=DemoScenes.Current,id=r.GetInstanceID()};')
            unity.evaluate(f'DemoScenes.Load({index}); return true;')
            wait_for(f'!DemoScenes.Loading && DemoScenes.Current=={index}')
            row=unity.evaluate(f'''
    var r=UnityEngine.Object.FindFirstObjectByType<CudaRenderer>();
    r.width=96; r.height=54; r.InputEnabled=false; r.backend=CudaRenderer.RenderBackend.Embree; r.RenderFrame();
    bool unloaded={str(previous['index']==index).lower()} || UnityEditor.EditorUtility.InstanceIDToObject({previous['id']})==null;
    return new {{scene=DemoScenes.Current,loaded=UnityEngine.SceneManagement.SceneManager.sceneCount,
        cameras=UnityEngine.Object.FindObjectsByType<CudaRenderer>(UnityEngine.FindObjectsSortMode.None).Length,
        unloaded=unloaded,samples=r.SampleCount,threads=r.CpuThreadCount,embreeBuildMs=r.EmbreeBuildMilliseconds,
        aiEnabled=r.denoise,aiMs=r.DenoiseMilliseconds,aiError=r.DenoiseError,displayedSamples=r.DisplayedSamples}};
    ''')
            assert row['loaded']==1 and row['cameras']==1 and row['unloaded'] and row['samples']>0 and row['threads']>1,row
            assert row['aiEnabled'] and row['aiMs']>0 and not row['aiError'] and 0<row['displayedSamples']<=row['samples'],row
            switches.append(row)
        (OUTPUT / 'switching.json').write_text(json.dumps(switches,indent=2)+'\n')
        print('PASS Play-mode switching across all demos and back; old cameras destroyed; Embree renders', flush=True)
finally:
    if original and original['playing']==playing:
        if playing:
            unity.evaluate(f"DemoScenes.Load({original['index']}); return true;")
            wait_for(f"!DemoScenes.Loading && DemoScenes.Current=={original['index']}")
            unity.evaluate(f"var r=UnityEngine.Object.FindFirstObjectByType<CudaRenderer>(); r.width={original['width']}; r.height={original['height']}; r.backend=(CudaRenderer.RenderBackend){original['backend']}; r.InputEnabled={str(original['input']).lower()}; return true;")
        else:
            unity.evaluate('UnityEditor.SceneManagement.EditorSceneManager.OpenScene('+json.dumps(original['path'])+'); return true;')
        unity.evaluate('CudaScenePreview.Enabled='+str(original['preview']).lower()+'; UnityEngine.Application.runInBackground='+str(original['background']).lower()+'; return true;')
    unity.close()

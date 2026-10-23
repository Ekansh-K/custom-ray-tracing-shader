// Differential BVH checks against a triangle loop, on both CPU and CUDA.
// This reference loop is only in this standalone verifier, never in the app.
#include "../Native/renderer.cu"
#include <cassert>
#include <iostream>
#include <random>

struct Query { float3 origin,direction; float limit; };
__host__ __device__ static Hit brute(const Query& q,Scene scene,bool any) {
    Hit result; result.distance=q.limit;
    for (int i=0;i<scene.triangleCount;i++) {
        float u,v,d=intersect(q.origin,q.direction,scene.triangles[i],u,v);
        if (d>0 && d<result.distance) {
            result.index=i; result.distance=d; result.u=u; result.v=v;
            if (any) return result;
        }
    }
    return result;
}
__host__ __device__ static bool equivalent(Hit a,Hit b,bool any) {
    if (any) return (a.index>=0)==(b.index>=0);
    if (a.index!=b.index) return false;
    return a.index<0 || (fabsf(a.distance-b.distance)<=1e-5f*fmaxf(1,fabsf(b.distance)) &&
                         fabsf(a.u-b.u)<=1e-5f && fabsf(a.v-b.v)<=1e-5f);
}
__global__ void checkQueries(const Query* queries,int count,Scene scene,int* errors) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=count) return;
    const Query& q=queries[i];
    for (bool any : {false,true}) {
        Hit expected=brute(q,scene,any),actual=collision(q.origin,q.direction,scene,q.limit,any);
        if (!equivalent(actual,expected,any)) atomicAdd(errors,1);
    }
}
static Triangle triangle(float3 a,float3 b,float3 c,int sided=0) {
    Triangle t{};
    t.origin=a; t.edge1=sub(b,a); t.edge2=sub(c,a); t.oneSided=sided;
    t.normal0=t.normal1=t.normal2=unit(cross(t.edge1,t.edge2));
    t.color=t.specularColor=make_float3(1,1,1);
    return t;
}
static void layout(const Session& s) {
    std::vector<int> visited(s.nodes.size()),covered(s.order.size());
    auto walk=[&](auto&& self,int index,int depth)->void {
        assert(index>=0 && index<(int)s.nodes.size() && !visited[index]++ && depth<=BvhMaxDepth);
        const BvhNode& n=s.nodes[index];
        if (n.count) {
            assert(n.first>=0 && n.first+n.count<=(int)s.order.size());
            for (int i=n.first;i<n.first+n.count;i++) {
                int id=s.order[i]; assert(id>=0 && id<(int)covered.size()); covered[id]++;
                const Triangle& t=s.triangles[id];
                for (float3 v : {t.origin,add(t.origin,t.edge1),add(t.origin,t.edge2)}) for (int a=0;a<3;a++)
                    assert((&v.x)[a]>=(&n.low.x)[a] && (&v.x)[a]<=(&n.high.x)[a]);
            }
        } else {
            for (int child=n.first;child<=n.first+1;child++) {
                assert(child>=0 && child<(int)s.nodes.size());
                for (int a=0;a<3;a++) assert((&s.nodes[child].low.x)[a]>=(&n.low.x)[a] && (&s.nodes[child].high.x)[a]<=(&n.high.x)[a]);
                self(self,child,depth+1);
            }
        }
    };
    walk(walk,0,0);
    for (int count:visited) assert(count==1);
    for (int count:covered) assert(count==1);
}
static int check(const char* name,const std::vector<Triangle>& triangles,const std::vector<Query>& queries,bool fullStack=false) {
    Session s;
    assert(RtInitialize(&s,triangles.data(),(int)triangles.size(),1,1));
    if (fullStack) {
        // A deliberate depth-32 chain exercises every stack slot, even though
        // the SAH builder normally chooses a much shallower tree for these faces.
        BvhNode box=s.nodes[0]; s.nodes.resize(1);
        int parent=0;
        for (int depth=0;depth<BvhMaxDepth;depth++) {
            int child=(int)s.nodes.size(); s.nodes.resize(s.nodes.size()+2);
            s.nodes[parent]={box.low,child,box.high,0};
            s.nodes[child+1]={box.low,depth,box.high,1}; parent=child;
        }
        s.nodes[parent]={box.low,BvhMaxDepth,box.high,1}; s.bvhDepth=BvhMaxDepth;
        cudaFree(s.deviceNodes); s.deviceNodes=nullptr;
        assert(upload(&s,s.deviceNodes,s.nodes));
    }
    layout(s);
    Scene host{s.triangles.data(),s.nodes.data(),s.order.data(),s.lights.data(),(int)s.triangles.size(),(int)s.lights.size(),s.lightArea};
    for (const Query& q:queries) for (bool any : {false,true})
        assert(equivalent(collision(q.origin,q.direction,host,q.limit,any),brute(q,host,any),any));
    Query* deviceQueries=nullptr; int* deviceErrors=nullptr;
    assert(cudaMalloc(&deviceQueries,queries.size()*sizeof(Query))==cudaSuccess);
    assert(cudaMalloc(&deviceErrors,sizeof(int))==cudaSuccess);
    assert(cudaMemcpy(deviceQueries,queries.data(),queries.size()*sizeof(Query),cudaMemcpyHostToDevice)==cudaSuccess);
    assert(cudaMemset(deviceErrors,0,sizeof(int))==cudaSuccess);
    Scene device{s.deviceTriangles,s.deviceNodes,s.deviceOrder,s.deviceLights,(int)s.triangles.size(),(int)s.lights.size(),s.lightArea};
    checkQueries<<<((int)queries.size()+127)/128,128>>>(deviceQueries,(int)queries.size(),device,deviceErrors);
    assert(cudaDeviceSynchronize()==cudaSuccess);
    int errors=0; assert(cudaMemcpy(&errors,deviceErrors,sizeof(int),cudaMemcpyDeviceToHost)==cudaSuccess);
    assert(errors==0);
    std::cout<<"PASS "<<name<<": "<<triangles.size()<<" triangles, "<<s.nodes.size()<<" nodes, depth "<<s.bvhDepth
             <<", "<<queries.size()<<" rays, nearest-hit and shadow tests on CPU/CUDA"<<std::endl;
    int depth=s.bvhDepth;
    cudaFree(deviceQueries); cudaFree(deviceErrors); RtShutdown(&s);
    return depth;
}
int main() {
    std::mt19937 rng(719393);
    std::uniform_real_distribution<float> value(-10,10);
    std::vector<Query> rays;
    for (int i=0;i<12000;i++) {
        float3 origin=make_float3(value(rng),value(rng),value(rng));
        float3 direction=unit(make_float3(value(rng),value(rng),value(rng)));
        if (i%4==0) direction=make_float3((i%3)==0,(i%3)==1,(i%3)==2);
        if (i%5==0) direction.x=1e-40f;
        rays.push_back({origin,direction,i%2 ? 1e30f : fabsf(value(rng))});
    }
    // Shared edges, duplicate triangles, one-sided faces, and zero-area inputs.
    std::vector<Triangle> edge{
        triangle(make_float3(-1,-1,0),make_float3(1,-1,0),make_float3(1,1,0)),
        triangle(make_float3(-1,-1,0),make_float3(1,1,0),make_float3(-1,1,0),1)};
    edge.push_back(edge[0]); edge.push_back(triangle(make_float3(0,0,0),make_float3(0,0,0),make_float3(0,0,0)));
    for (float z : {-1.0f,0.0f,1.0f}) for (float x : {-1.0f,0.0f,1.0f})
        for (float3 d : {make_float3(0,0,1),make_float3(0,0,-1),make_float3(0,1,0),make_float3(1e-40f,0,1)})
            for (float limit : {.5f,1.0f,2.0f,1e30f}) rays.push_back({make_float3(x,x,z),d,limit});
    check("edges, ties, parallel, inside, finite limits",edge,rays);
    std::vector<Triangle> scatter;
    for (int i=0;i<3000;i++) {
        float3 a=make_float3(value(rng),value(rng),value(rng));
        scatter.push_back(triangle(a,add(a,make_float3(.2f,.3f,0)),add(a,make_float3(0,.1f,.4f)),i%2));
    }
    check("scattered geometry",scatter,rays);
    std::vector<Triangle> overlap(256,edge[0]);
    check("identical centroids and overlapping faces",overlap,rays);
    std::vector<Triangle> deep;
    for (int i=0;i<200;i++) {
        float x=powf(1.5f,(float)i);
        if (!std::isfinite(x)) break;
        deep.push_back(triangle(make_float3(x,-1,0),make_float3(x,1,0),make_float3(x,0,1)));
    }
    check("unbalanced spatial distribution",deep,rays);
    assert(check("full depth-32 traversal stack",std::vector<Triangle>(BvhMaxDepth+1,edge[0]),rays,true)==BvhMaxDepth);
    std::cout<<"All BVH structure and differential checks passed.\n";
}

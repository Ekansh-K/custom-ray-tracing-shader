#pragma once
#include <algorithm>
#include <cfloat>
#include <numeric>

// Adjacent child nodes and 32-byte records keep traversal reads compact.
struct BvhNode { float3 low; int first; float3 high; int count; };
static_assert(sizeof(BvhNode)==32);
constexpr int BvhMaxDepth=32;

struct BvhBounds {
    float3 low=make_float3(FLT_MAX,FLT_MAX,FLT_MAX), high=make_float3(-FLT_MAX,-FLT_MAX,-FLT_MAX);
    void grow(float3 a,float3 b) {
        for (int axis=0;axis<3;axis++) {
            (&low.x)[axis]=fminf((&low.x)[axis],(&a.x)[axis]);
            (&high.x)[axis]=fmaxf((&high.x)[axis],(&b.x)[axis]);
        }
    }
    void grow(const BvhBounds& other) { grow(other.low,other.high); }
    float area() const {
        float x=fmaxf(0,high.x-low.x),y=fmaxf(0,high.y-low.y),z=fmaxf(0,high.z-low.z);
        return x*y+x*z+y*z; // Half the surface area; the common factor cancels.
    }
};

struct BvhBuilder {
    struct Primitive { BvhBounds bounds; float3 center; };
    struct Bin { BvhBounds bounds; int count=0; };
    static constexpr int BinCount=12;
    std::vector<Primitive> primitives;
    std::vector<int> order;
    std::vector<BvhNode> nodes;
    int maxDepth=0;

    explicit BvhBuilder(const std::vector<Triangle>& triangles) {
        primitives.resize(triangles.size()); order.resize(triangles.size());
        std::iota(order.begin(),order.end(),0);
        for (size_t i=0;i<triangles.size();i++) {
            const Triangle& t=triangles[i];
            float3 b=make_float3(t.origin.x+t.edge1.x,t.origin.y+t.edge1.y,t.origin.z+t.edge1.z);
            float3 c=make_float3(t.origin.x+t.edge2.x,t.origin.y+t.edge2.y,t.origin.z+t.edge2.z);
            auto& p=primitives[i];
            p.bounds.grow(t.origin,t.origin); p.bounds.grow(b,b); p.bounds.grow(c,c);
            for (int a=0;a<3;a++) {
                (&p.center.x)[a]=((&t.origin.x)[a]+(&b.x)[a]+(&c.x)[a])/3;
                float padding=1e-5f+1e-6f*fmaxf(fabsf((&p.bounds.low.x)[a]),fabsf((&p.bounds.high.x)[a]));
                (&p.bounds.low.x)[a]-=padding; (&p.bounds.high.x)[a]+=padding;
            }
        }
        nodes.reserve(triangles.size()*2); nodes.resize(1);
        split(0,0,(int)triangles.size(),0);
    }
    static int bin(float center,float minimum,float scale) {
        return std::min(BinCount-1,std::max(0,(int)((center-minimum)*scale)));
    }
    void split(int nodeIndex,int first,int count,int depth) {
        maxDepth=std::max(maxDepth,depth);
        BvhBounds bounds,centers;
        for (int i=first;i<first+count;i++) {
            const auto& p=primitives[order[i]]; bounds.grow(p.bounds); centers.grow(p.center,p.center);
        }
        nodes[nodeIndex]={bounds.low,first,bounds.high,count};
        // The depth limit is also the traversal stack's capacity. Never truncate rays.
        if (count<=4 || depth==BvhMaxDepth) return;
        int bestAxis=-1,bestBin=-1;
        float bestCost=bounds.area()*count;
        for (int axis=0;axis<3;axis++) {
            float extent=(&centers.high.x)[axis]-(&centers.low.x)[axis];
            if (extent<=1e-12f) continue;
            float scale=BinCount/extent,minimum=(&centers.low.x)[axis];
            Bin bins[BinCount];
            for (int i=first;i<first+count;i++) {
                const auto& p=primitives[order[i]];
                Bin& bucket=bins[bin((&p.center.x)[axis],minimum,scale)];
                bucket.count++; bucket.bounds.grow(p.bounds);
            }
            BvhBounds rightBounds[BinCount],right;
            int rightCounts[BinCount],rightCount=0;
            for (int i=BinCount-1;i>=0;i--) {
                if (bins[i].count) right.grow(bins[i].bounds);
                rightCount+=bins[i].count; rightBounds[i]=right; rightCounts[i]=rightCount;
            }
            BvhBounds left;
            int leftCount=0;
            for (int i=0;i<BinCount-1;i++) {
                if (bins[i].count) left.grow(bins[i].bounds);
                leftCount+=bins[i].count;
                if (!leftCount || !rightCounts[i+1]) continue;
                float cost=bounds.area()+left.area()*leftCount+rightBounds[i+1].area()*rightCounts[i+1];
                if (cost<bestCost) { bestCost=cost; bestAxis=axis; bestBin=i; }
            }
        }
        if (bestAxis<0) return; // Keeping a leaf can be cheaper than another split.
        float minimum=(&centers.low.x)[bestAxis];
        float scale=BinCount/((&centers.high.x)[bestAxis]-minimum);
        auto middle=std::partition(order.begin()+first,order.begin()+first+count,[&](int i) {
            return bin((&primitives[i].center.x)[bestAxis],minimum,scale)<=bestBin;
        });
        int leftCount=(int)(middle-(order.begin()+first));
        if (leftCount==0 || leftCount==count) return;
        int child=(int)nodes.size(); nodes.resize(nodes.size()+2);
        nodes[nodeIndex].first=child; nodes[nodeIndex].count=0;
        split(child,first,leftCount,depth+1);
        split(child+1,first+leftCount,count-leftCount,depth+1);
    }
};

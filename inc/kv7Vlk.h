#ifndef __kv7Vlk_H_INCLUDED_2021_08_04__
#define __kv7Vlk_H_INCLUDED_2021_08_04__

/* Note ----------------------
* kmMat has been created by Choi, Kiwan
* This is version 7
* kmMat v7 is including the following
*   - km7Define.h
*   - km7Define.h -> km7Mat.h
*   - km7Define.h -> km7Mat.h -> km7Wnd.h
*   - km7Define.h -> km7Mat.h -> kc7Mat.h -> km7Dnn.h
*   - km7Define.h -> km7Mat.h -> kc7Mat.h
*   - km7Define.h -> km7Mat.h -> km7Net.h
*   - km7Define.h -> km7Mat.h -> km7Wnd.h --> kv7Vlk.h
*/

// base header
#include "km7Mat.h"

#include <fstream>
#include <vector>
#include <array>

// define for vulkan's setting
#define VK_USE_PLATFORM_WIN32_KHR

// vulkan header
#include <vulkan/vulkan.h>           // $(VK_SDK_PATH)/include
#include <glm/glm.hpp>               // you must copy glm folder to $(VK_SDK_PATH)/include

#pragma comment(lib,"vulkan-1.lib")  // $(VK_SDK_PATH)/lib

///////////////////////////////////////////////////////////////
// vertex stucture

struct vertex
{
	glm::vec2 pos;
	glm::vec3 color;

	static VkVertexInputBindingDescription GetBndDsc()
	{
		VkVertexInputBindingDescription bnd_dsc{};
		bnd_dsc.binding   = 0;
		bnd_dsc.stride    = sizeof(vertex);
		bnd_dsc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		return bnd_dsc;
	};

	static kmMat1<VkVertexInputAttributeDescription> GetAttrDscs()
	{
		kmMat1<VkVertexInputAttributeDescription> attr_dsc(2); attr_dsc.SetZero();

		attr_dsc(0).binding  = 0;
		attr_dsc(0).location = 0;
		attr_dsc(0).format   = VK_FORMAT_R32G32_SFLOAT;
		attr_dsc(0).offset   = offsetof(vertex, pos);

		attr_dsc(1).binding  = 0;
		attr_dsc(1).location = 1;
		attr_dsc(1).format   = VK_FORMAT_R32G32B32_SFLOAT;
		attr_dsc(1).offset   = offsetof(vertex, color);

		return attr_dsc;
	};
};

///////////////////////////////////////////////////////////////
// base class for vulkan
class kvVlk
{
protected:
	VkInstance       _inst    {};  // instance of vulkan
	VkPhysicalDevice _pdv     {};  // physical device
	VkDevice         _ldv     {};  // logical device
	VkQueue          _que     {};  // queue
	VkSurfaceKHR     _srf     {};  // window surface
	VkSwapchainKHR   _swp     {};  // swap chain
	VkCommandPool    _cmdp    {};  // command pool
	VkCommandBuffer  _cbuf    {};  // command buffer
	VkFence          _fnc     {};  // fence...      AcquireNextImg() is waiting for OS's work
	VkSemaphore      _smp0    {};  // semaphore ... SubmitQue()  is waiting for AcquireNextImg()
	VkSemaphore      _smp1    {};  // semaphore ... PresentImg() is waiting for SubmitQue();
	VkShaderModule   _shd_vert{};  // shader module for vertex
	VkShaderModule   _shd_frag{};  // shader module for fragment
	VkRenderPass     _rndpass {};  // render pass
	VkPipelineLayout _ppllyout{};  // pipeline layout
	VkPipeline       _ppl     {};  // pipeline

	int                        _quef_idx = 0;   // queue family index
	uint                       _swp_idx  = 0;   // current swap chain image index
	VkFormat                   _swp_fmt{};      // swap chain format
	VkExtent2D                 _swp_ext{};      // swap chain extent

	uint                       _imgs_n;         // number of swap chain images
	kmMat1<VkImage>            _imgs;           // swap chain images
	kmMat1<VkImageView>        _views;          // swap chain image views
	kmMat1<VkFramebuffer>      _fbufs;          // swap chain frame buffers

	VkPhysicalDeviceProperties _pdv_prop{};     // physical device properties
	VkPhysicalDeviceFeatures   _pdv_feat{};     // physical device features

	VkBuffer       _vert_buf    {};   // vertex buffer
	VkDeviceMemory _vert_buf_mem{};   // vertex buffer's memory

public:
	// constructor
	kvVlk() {};
	kvVlk(HWND hwnd) { Init(hwnd); };

	// destructor
	virtual ~kvVlk() { Destroy(); };

	////////////////////////////////////////
	// init vulkan

	// init vulkan
	kvVlk& Init(HWND hwnd, int disp_on = 0)
	{
		print("\n* ------------------------------ Init() starts\n");

		// display available instance extensions and layers
		if(disp_on)
		{
			kvVlk::DisplayInstExtns(); 
			kvVlk::DisplayInstLyrs ();
		}

		// create instance... _inst
		CreateInst();

		// get physical device... _pdv
		GetPdv(1); 

		// display available device extensions and layers
		if(disp_on)
		{
			DisplayPdvExtns();
			DisplayPdvLyrs ();
		}

		// get index of queue fimily of physical device... _quef_idx
		GetQuefIdx();

		// create logical device... _ldv
		CreateLdv();

		// get queue handle... _que
		GetQue();

		// create window suface .. _srf
		CreateSrf(hwnd);

		if(IsSupportedSrf() == VK_TRUE) print("* srf is supported\n");
		else                            print("* srf is not supported\n");

		// display surface mode
		if(disp_on)
		{
			DisplayPresentModes();
			DisplaySrfFormats  ();
		}

		// create swap chain... _swp
		CreateSwp();

		// get swapchain images... _imgs
		GetSwpImg();

		// create swap chain views... _views
		CreateViews();

		// create command pool... _cmdp
		CreateCmdp();

		// allocate command buffer... _cbuf
		AllocateCbuf();

		// create render pass... _rndpass
		CreateRndPass();

		// create frame buffers... _fbufs
		CreateFbufs();

		// create fence... _fnc
		CreateFnc();

		// creaet semaphore... _smp0, _smp1
		CreateSmp();

		// change image lyout to present src
		BeginCbuf();
		ChangeLyoutAll(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
		EndCbuf().SubmitQueNoSync().WaitLdvIdle();

		print("* ------------------------------ Init() ends\n\n");

		return *this;
	};
	
	// create instance... _inst
	kvVlk& CreateInst()
	{
		// set application info
		VkApplicationInfo app{};

		app.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO;
		app.pApplicationName   = "kv7Vlk";
		app.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
		app.pEngineName        = "No Engine";
		app.engineVersion      = VK_MAKE_VERSION(1, 0, 0);
		app.apiVersion         = VK_API_VERSION_1_0;

		// set layers
		const char* lyrs[] = { "VK_LAYER_KHRONOS_validation" };

		// set extensions
		const char* extns[] = { "VK_KHR_surface", "VK_KHR_win32_surface" };	

		// set instance creating info
		VkInstanceCreateInfo cinfo{};

		cinfo.sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
		cinfo.pApplicationInfo        = &app;
		cinfo.enabledLayerCount       = numof(lyrs);
		cinfo.ppEnabledLayerNames     = lyrs;
		cinfo.enabledExtensionCount   = numof(extns);
		cinfo.ppEnabledExtensionNames = extns;

		// create instance
		VkResult res = vkCreateInstance(&cinfo, nullptr, &_inst);

		CheckRes(res, "creating instance");	return *this;
	};
	
	// get physical device... _pdv
	kvVlk& GetPdv(int idx = 0)
	{
		// get physical devices
		uint pdv_n; 
		vkEnumeratePhysicalDevices(_inst, &pdv_n, nullptr);

		kmMat1<VkPhysicalDevice> pdv_list(pdv_n);
		vkEnumeratePhysicalDevices(_inst, &pdv_n, pdv_list.P());

		// check physical device suitability 
		VkPhysicalDeviceProperties pdv_prop;
		VkPhysicalDeviceFeatures   pdv_feat;

		print("* number of physical devices : %d\n", pdv_n);

		for(uint i = 0; i < pdv_n; ++i)
		{
			vkGetPhysicalDeviceProperties(pdv_list(i), &pdv_prop);
			vkGetPhysicalDeviceFeatures  (pdv_list(i), &pdv_feat);

			print("* (%d) %s", i, pdv_prop.deviceName);
			print(" (id : %d, type : %d", pdv_prop.deviceID, pdv_prop.deviceType);
			print(", feature : %d)\n", pdv_feat.geometryShader);
		}
		// select device 
		_pdv = pdv_list(idx);

		vkGetPhysicalDeviceProperties(_pdv, &_pdv_prop);
		vkGetPhysicalDeviceFeatures  (_pdv, &_pdv_feat);

		print("* selected physical device : (%d) %s\n\n", idx, _pdv_prop.deviceName);

		return *this;
	};

	// get index of queue family... _quef_idx
	kvVlk& GetQuefIdx()
	{
		// get list of queue family		
		uint quef_n = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(_pdv, &quef_n, nullptr);

		kmMat1<VkQueueFamilyProperties> quef_list(quef_n);
		vkGetPhysicalDeviceQueueFamilyProperties(_pdv, &quef_n, quef_list.P());

		// display list of queue family
		int quef_idx = 0;

		print("* number of queue family : %d\n", quef_n);
		for(uint i = 0; i < quef_n; ++i)
		{
			const int graphics_bit = quef_list(i).queueFlags & VK_QUEUE_GRAPHICS_BIT;

			print("* (%d) flag %d (graphics : %s), count %d\n", i, quef_list(i).queueFlags, 
				(graphics_bit)?"on ":"off", quef_list(i).queueCount);

			if(graphics_bit > 0) quef_idx = i;
		}
		print("* selected index of queue family : %d\n\n", quef_idx);

		// set index of queue family
		_quef_idx = quef_idx;

		return *this;
	};

	// create logical device... _ldv
	kvVlk& CreateLdv()
	{
		// set device queue creating info
		float que_priority = 1.f;  // 0 - 1.f (0 : low priority, 1 : high priority)

		VkDeviceQueueCreateInfo quecinfo{};
		quecinfo.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		quecinfo.queueFamilyIndex = _quef_idx;
		quecinfo.queueCount       = 1;
		quecinfo.pQueuePriorities = &que_priority;

		// set extensions
		const char* extns[] = { "VK_KHR_swapchain" }; 		

		// set device creating info
		VkDeviceCreateInfo ldvcinfo{};

		ldvcinfo.sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
		ldvcinfo.pQueueCreateInfos       = &quecinfo;
		ldvcinfo.queueCreateInfoCount    = 1;
		ldvcinfo.pEnabledFeatures        = &_pdv_feat;
		ldvcinfo.enabledLayerCount       = 0;
		ldvcinfo.enabledExtensionCount   = numof(extns);
		ldvcinfo.ppEnabledExtensionNames = extns;		

		// create logical device
		VkResult res = vkCreateDevice(_pdv, &ldvcinfo, nullptr, &_ldv);

		CheckRes(res, "creating logical device"); return *this;
	};

	// get device queue... _que
	kvVlk& GetQue()
	{
		// get device queue
		vkGetDeviceQueue(_ldv, _quef_idx, 0, &_que);

		return *this;
	};

	// create window surface... _srf
	// * Note that this needs instance extensions (VK_KHR_surface, VK_KHR_win32_surface).
	kvVlk& CreateSrf(const HWND hwnd)
	{
		// set surface creating info
		VkWin32SurfaceCreateInfoKHR srf_cinfo{};
		srf_cinfo.sType     = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
		srf_cinfo.hwnd      = hwnd;
		srf_cinfo.hinstance = GetModuleHandle(nullptr);

		// create surface
		VkResult res = vkCreateWin32SurfaceKHR(_inst, &srf_cinfo, nullptr, &_srf);

		CheckRes(res, "creating surface"); return *this;
	};

	// create swap chain... _swp
	// * Note that this needs device extension (VK_KHR_swapchain)
	kvVlk& CreateSwp(uint imgs_n = 3)
	{
		// get surface capabilities
		VkSurfaceCapabilitiesKHR srf_capa;
		vkGetPhysicalDeviceSurfaceCapabilitiesKHR(_pdv, _srf, &srf_capa);

		// get surface format
		VkSurfaceFormatKHR srf_format = GetSrfFormat(0);

		// set swap chain creating info
		VkSwapchainCreateInfoKHR swp_cinfo{};
		swp_cinfo.sType            = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
		swp_cinfo.surface          = _srf;
		swp_cinfo.minImageCount    = _imgs_n = imgs_n;
		swp_cinfo.imageColorSpace  =               srf_format.colorSpace;
		swp_cinfo.imageFormat      = _swp_fmt = srf_format.format;
		swp_cinfo.imageExtent      = _swp_ext = srf_capa.currentExtent;
		swp_cinfo.imageArrayLayers = 1;
		swp_cinfo.imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
		swp_cinfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
		swp_cinfo.preTransform     = srf_capa.currentTransform;
		swp_cinfo.compositeAlpha   = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
		swp_cinfo.presentMode      = GetPresentMode(0);

		// create swapchain
		VkResult res = vkCreateSwapchainKHR(_ldv, &swp_cinfo, nullptr, &_swp);

		CheckRes(res, "creating swapchain"); return *this;
	};

	// get swap chain images... _imgs
	kvVlk& GetSwpImg()
	{
		// get swap chain images		
		vkGetSwapchainImagesKHR(_ldv, _swp, &_imgs_n, nullptr);

		_imgs.RecreateIf(_imgs_n);
		VkResult res = vkGetSwapchainImagesKHR(_ldv, _swp, &_imgs_n, _imgs.P());

		CheckRes(res, "getting swapchain images"); return *this;
	};

	// creaate swap chain image views... _views
	kvVlk& CreateViews()
	{
		_views.RecreateIf(_imgs_n);

		for(uint i = 0; i < _imgs_n; ++i)
		{
			// set image view creating info
			VkImageViewCreateInfo view_cinfo{};
			view_cinfo.sType        = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
			view_cinfo.image        = _imgs(i);
			view_cinfo.viewType     = VK_IMAGE_VIEW_TYPE_2D;
			view_cinfo.format       = _swp_fmt;
			view_cinfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
			view_cinfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
			view_cinfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
			view_cinfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
			view_cinfo.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

			// create image view
			auto res = vkCreateImageView(_ldv, &view_cinfo, nullptr, &_views(i));

			CheckRes(res, "creating view");
		}
		return *this;
	};

	// create command pool... _cmdp
	kvVlk& CreateCmdp()
	{
		// set command pool creating info
		VkCommandPoolCreateInfo cmdp_cinfo{};
		cmdp_cinfo.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		cmdp_cinfo.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		cmdp_cinfo.queueFamilyIndex = _quef_idx;

		// create command pool
		VkResult res = vkCreateCommandPool(_ldv, &cmdp_cinfo, nullptr, &_cmdp);

		CheckRes(res, "creating command pool"); return *this;
	};

	// allocate command buffer... _cbuf
	kvVlk& AllocateCbuf()
	{
		// set command buffer allocating info
		VkCommandBufferAllocateInfo cbuf_ainfo{};
		cbuf_ainfo.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		cbuf_ainfo.commandPool        = _cmdp;
		cbuf_ainfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		cbuf_ainfo.commandBufferCount = 1;

		// allocate command buffer
		VkResult res = vkAllocateCommandBuffers(_ldv, &cbuf_ainfo, &_cbuf);

		CheckRes(res, "allocating command buffer"); return *this;
	};

	// create fence... _fnc
	kvVlk& CreateFnc()
	{
		// set fence creating info
		VkFenceCreateInfo fnc_cinfo{};
		fnc_cinfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

		// create fence
		VkResult res = vkCreateFence(_ldv, &fnc_cinfo, nullptr, &_fnc);

		CheckRes(res, "creating fence"); return *this;
	};

	// create semaphore... _smp
	kvVlk& CreateSmp()
	{
		// set semaphore creating info
		VkSemaphoreCreateInfo smp_cinfo{};
		smp_cinfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

		// create semaphore
		VkResult res;
		res = vkCreateSemaphore(_ldv, &smp_cinfo, nullptr, &_smp0); CheckRes(res, "creating semaphore for acquiring");
		res = vkCreateSemaphore(_ldv, &smp_cinfo, nullptr, &_smp1); CheckRes(res, "creating semaphore for rendering");

		return *this;
	};

	// destroy
	kvVlk& Destroy()
	{
		for(int i = 0; i < _fbufs.N(); ++i) vkDestroyFramebuffer( _ldv, _fbufs(i), nullptr);

		vkDestroyPipeline      ( _ldv, _ppl     , nullptr);
		vkDestroyPipelineLayout( _ldv, _ppllyout, nullptr);
		vkDestroyRenderPass    ( _ldv, _rndpass , nullptr);
		vkDestroySemaphore     ( _ldv, _smp1    , nullptr);
		vkDestroySemaphore     ( _ldv, _smp0    , nullptr);
		vkDestroyFence         ( _ldv, _fnc     , nullptr);
		vkDestroyCommandPool   ( _ldv, _cmdp    , nullptr);

		for(int i = 0; i < _views.N(); ++i) vkDestroyImageView( _ldv,  _views(i), nullptr);

		vkDestroySwapchainKHR( _ldv ,     _swp, nullptr);
		vkDestroyBuffer      ( _ldv, _vert_buf, nullptr); vkFreeMemory( _ldv, _vert_buf_mem, nullptr);
		vkDestroySurfaceKHR  ( _inst,     _srf, nullptr);
		vkDestroyDevice      ( _ldv ,  nullptr);
		vkDestroyInstance    ( _inst,  nullptr);

		return *this;
	};

	////////////////////////////////////////
	// shader functions
	
	// load vertex file and create shader module
	kvVlk& LoadVert(const char* file_name) { auto code = ReadFile(file_name); return CreateShd(code, _shd_vert); };
	kvVlk& LoadFrag(const char* file_name) { auto code = ReadFile(file_name); return CreateShd(code, _shd_frag); };

	// create shader module
	kvVlk& CreateShd(const kmMat1i8& code, VkShaderModule& shd)
	{
		VkShaderModuleCreateInfo shd_cinfo{};
		shd_cinfo.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		shd_cinfo.codeSize = code.N();
		shd_cinfo.pCode    = (uint*)code.P();

		VkResult res = vkCreateShaderModule(_ldv, &shd_cinfo, nullptr, &shd);

		CheckRes(res, "creating shader module"); return *this;
	};

	// create render pass... _rndpass
	kvVlk& CreateRndPass()
	{
		// set attachment description... color_dsc
		VkAttachmentDescription att_dsc{};
		att_dsc.format         = _swp_fmt;
		att_dsc.samples        = VK_SAMPLE_COUNT_1_BIT;
		att_dsc.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
		att_dsc.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
		att_dsc.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		att_dsc.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		att_dsc.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
		att_dsc.finalLayout    = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

		// set attachment reference... color_ref
		VkAttachmentReference att_ref{};
		att_ref.attachment = 0;
		att_ref.layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		// set subpass description... subpass_dsc
		VkSubpassDescription subpass_dsc{};
		subpass_dsc.pipelineBindPoint    = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpass_dsc.colorAttachmentCount = 1;
		subpass_dsc.pColorAttachments    = &att_ref;

		// set subpass dependency... subpass_dpn
		VkSubpassDependency subpass_dpn{};
		subpass_dpn.srcSubpass    = VK_SUBPASS_EXTERNAL;
		subpass_dpn.dstSubpass    = 0;
		subpass_dpn.srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		subpass_dpn.srcAccessMask = 0;
		subpass_dpn.dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		subpass_dpn.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

		//set render pass creating info... rndpass_cinfo
		VkRenderPassCreateInfo rndpass_cinfo{};
		rndpass_cinfo.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		rndpass_cinfo.attachmentCount = 1;
		rndpass_cinfo.pAttachments    = &att_dsc;
		rndpass_cinfo.subpassCount    = 1;
		rndpass_cinfo.pSubpasses      = &subpass_dsc;
		rndpass_cinfo.dependencyCount = 1;
		rndpass_cinfo.pDependencies   = &subpass_dpn;

		// create render pass... _rndpass
		VkResult res = vkCreateRenderPass(_ldv, &rndpass_cinfo, nullptr, &_rndpass);

		CheckRes(res, "creating render pass"); return *this;
	};

	// create graphics pipeline... _ppl
	kvVlk& CreateGraphicsPipeline(int vertin_mode = 0)
	{
		// set shader stage creating info... stg_cinfo
		VkPipelineShaderStageCreateInfo stg_vert_cinfo{};
		stg_vert_cinfo.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stg_vert_cinfo.stage  = VK_SHADER_STAGE_VERTEX_BIT;
		stg_vert_cinfo.module = _shd_vert;
		stg_vert_cinfo.pName  = "main";

		VkPipelineShaderStageCreateInfo stg_frag_cinfo{};
		stg_frag_cinfo.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stg_frag_cinfo.stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
		stg_frag_cinfo.module = _shd_frag;
		stg_frag_cinfo.pName  = "main";

		VkPipelineShaderStageCreateInfo stg_cinfo[] = { stg_vert_cinfo, stg_frag_cinfo };

		// set vertex input state creating info... vertin_cinfo
		VkPipelineVertexInputStateCreateInfo vertin_cinfo{};
		vertin_cinfo.sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		vertin_cinfo.vertexBindingDescriptionCount   = 0;
		vertin_cinfo.vertexAttributeDescriptionCount = 0;

		VkVertexInputBindingDescription           bnd_dsc;
		kmMat1<VkVertexInputAttributeDescription> attr_dsc;
		
		if(vertin_mode == 1)
		{
			bnd_dsc  = vertex::GetBndDsc  ();
			attr_dsc = vertex::GetAttrDscs();

			vertin_cinfo.vertexBindingDescriptionCount   = 1;
			vertin_cinfo.vertexAttributeDescriptionCount = 2;
			vertin_cinfo.pVertexBindingDescriptions      = &bnd_dsc;
			vertin_cinfo.pVertexAttributeDescriptions    = attr_dsc.P();
		}

		// set input assmbly state creating info... inasm_cinfo
		VkPipelineInputAssemblyStateCreateInfo inasm_cinfo{};
		inasm_cinfo.sType                  = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		inasm_cinfo.topology               = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		inasm_cinfo.primitiveRestartEnable = VK_FALSE;

		// set view port... vport
		VkViewport vport{};
		vport.x        = 0.f;
		vport.y        = 0.f;
		vport.width    = (float)_swp_ext.width;
		vport.height   = (float)_swp_ext.height;
		vport.minDepth = 0.f;
		vport.maxDepth = 1.f;

		// set scissor... scissor
		VkRect2D scissor{};
		scissor.offset = {0, 0};
		scissor.extent = _swp_ext;

		// set view port state creating info... vp_cinfo
		VkPipelineViewportStateCreateInfo vp_cinfo{};
		vp_cinfo.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		vp_cinfo.viewportCount = 1;
		vp_cinfo.pViewports    = &vport;
		vp_cinfo.scissorCount  = 1;
		vp_cinfo.pScissors     = &scissor;

		// set rasterization state creating info... rst_cinfo
		VkPipelineRasterizationStateCreateInfo rst_cinfo{};
		rst_cinfo.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rst_cinfo.depthClampEnable        = VK_FALSE;
		rst_cinfo.rasterizerDiscardEnable = VK_FALSE;
		rst_cinfo.polygonMode             = VK_POLYGON_MODE_FILL;
		rst_cinfo.lineWidth               = 1.f;
		rst_cinfo.cullMode                = VK_CULL_MODE_BACK_BIT;
		rst_cinfo.frontFace               = VK_FRONT_FACE_CLOCKWISE;
		rst_cinfo.depthBiasEnable         = VK_FALSE;

		// set multi-sample state creating info... multi_cinfo
		VkPipelineMultisampleStateCreateInfo multi_cinfo{};
		multi_cinfo.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multi_cinfo.sampleShadingEnable  = VK_FALSE;
		multi_cinfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

		// set color blend attachment state... color_att
		VkPipelineColorBlendAttachmentState color_att{};
		color_att.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		color_att.blendEnable    = VK_FALSE;

		// set color blend state creating info... color_cinfo
		VkPipelineColorBlendStateCreateInfo color_cinfo{};
		color_cinfo.sType             = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		color_cinfo.logicOpEnable     = VK_FALSE;
		color_cinfo.logicOp           = VK_LOGIC_OP_COPY; // optional
		color_cinfo.attachmentCount   = 1;
		color_cinfo.pAttachments      = &color_att;
		color_cinfo.blendConstants[0] = 0.f;  // optional
		color_cinfo.blendConstants[1] = 0.f;  // optional
		color_cinfo.blendConstants[2] = 0.f;  // optional
		color_cinfo.blendConstants[3] = 0.f;  // optional

		// set pipeline layout creating info... pplyout_cinfo
		VkPipelineLayoutCreateInfo pplyout_cinfo{};
		pplyout_cinfo.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pplyout_cinfo.setLayoutCount         = 0;
		pplyout_cinfo.pushConstantRangeCount = 0;

		// create pipeline layout... _pplyout
		VkResult res = vkCreatePipelineLayout(_ldv, &pplyout_cinfo, nullptr, &_ppllyout);

		CheckRes(res, "creating pipeline layout");

		// set pipeline creaating info... ppl_cinfo
		VkGraphicsPipelineCreateInfo ppl_cinfo{};
		ppl_cinfo.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		ppl_cinfo.stageCount          = 2;
		ppl_cinfo.pStages             = stg_cinfo;
		ppl_cinfo.pVertexInputState   = &vertin_cinfo;
		ppl_cinfo.pInputAssemblyState = &inasm_cinfo;
		ppl_cinfo.pViewportState      = &vp_cinfo;
		ppl_cinfo.pRasterizationState = &rst_cinfo;
		ppl_cinfo.pMultisampleState   = &multi_cinfo;
		ppl_cinfo.pDepthStencilState  = nullptr;             // optional
		ppl_cinfo.pColorBlendState    = &color_cinfo;
		ppl_cinfo.pDynamicState       = nullptr;             // optional
		ppl_cinfo.layout              = _ppllyout;
		ppl_cinfo.renderPass          = _rndpass;
		ppl_cinfo.subpass             = 0;
		ppl_cinfo.basePipelineHandle  = VK_NULL_HANDLE;      //optional
		ppl_cinfo.basePipelineIndex   = -1;                  // optional

		// create graphics pipeline
		res = vkCreateGraphicsPipelines(_ldv, VK_NULL_HANDLE, 1, &ppl_cinfo, nullptr, &_ppl);

		CheckRes(res, "creating graphics pipelines");

		// destroy shader module
		vkDestroyShaderModule( _ldv, _shd_frag, nullptr);
		vkDestroyShaderModule( _ldv, _shd_vert, nullptr);

		return *this;
	};

	// create swap chain frame buffers... _fbufs
	kvVlk& CreateFbufs()
	{
		_fbufs.RecreateIf(_imgs_n);

		for(uint i = 0; i < _imgs_n; ++i)
		{
			VkImageView att[] = { _views(i) };

			VkFramebufferCreateInfo frame_cinfo{};
			frame_cinfo.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			frame_cinfo.renderPass      = _rndpass;
			frame_cinfo.attachmentCount = 1;
			frame_cinfo.pAttachments    = att;
			frame_cinfo.width           = _swp_ext.width;
			frame_cinfo.height          = _swp_ext.height;
			frame_cinfo.layers          = 1;

			VkResult res = vkCreateFramebuffer(_ldv, &frame_cinfo, nullptr, &_fbufs(i));

			CheckRes(res, "createing frame buffers");
		}
		return *this;
	};

	////////////////////////////////////////
	// drawing ready functions

	// start command buffer
	kvVlk& BeginCbuf()
	{
		// set command buffer begining info
		VkCommandBufferBeginInfo cbuf_binfo{};
		cbuf_binfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		cbuf_binfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

		// begin command buffer
		VkResult res = vkBeginCommandBuffer(_cbuf, &cbuf_binfo);

		CheckRes(res, "begining command buffer"); return *this;
	};

	// end command buffer
	kvVlk& EndCbuf()
	{
		VkResult res = vkEndCommandBuffer(_cbuf);

		CheckRes(res, "ending command buffer"); return *this;
	};

	// start render pass
	kvVlk& BeginRndPass()
	{
		VkClearValue clear_color = {0.f, 0.f, 0.f, 1.f};

		// set render pass begining info
		VkRenderPassBeginInfo rndpass_binfo{};
		rndpass_binfo.sType             = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		rndpass_binfo.renderPass        = _rndpass;
		rndpass_binfo.framebuffer       = _fbufs(_swp_idx);
		rndpass_binfo.renderArea.offset = {0,0};
		rndpass_binfo.renderArea.extent = _swp_ext;
		rndpass_binfo.clearValueCount   = 1;
		rndpass_binfo.pClearValues      = &clear_color;

		// begin render pass
		vkCmdBeginRenderPass(_cbuf, &rndpass_binfo, VK_SUBPASS_CONTENTS_INLINE);

		print("* begin render pass : frame buf(%d)\n", _swp_idx);

		return *this;
	};

	// end render pass
	kvVlk& EndRndPass() { vkCmdEndRenderPass(_cbuf); return *this; };

	// submit queue without sync
	kvVlk& SubmitQueNoSync()
	{	
		// set submit info
		VkSubmitInfo sinfo{};
		sinfo.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		sinfo.commandBufferCount = 1;
		sinfo.pCommandBuffers    = &_cbuf;

		// submit queue
		VkResult res = vkQueueSubmit(_que, 1, &sinfo, VK_NULL_HANDLE);

		CheckRes(res, "submitting queue no sync"); return *this;
	}; 

	// submit queue
	kvVlk& SubmitQue()
	{
		// set pipeline stage flags
		VkPipelineStageFlags stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

		// set submit info
		VkSubmitInfo sinfo{};
		sinfo.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		sinfo.commandBufferCount = 1;
		sinfo.pCommandBuffers    = &_cbuf;
		sinfo.waitSemaphoreCount = 1;
		sinfo.pWaitSemaphores    = &_smp0;
		sinfo.pWaitDstStageMask  = &stage;
		sinfo.signalSemaphoreCount = 1;
		sinfo.pSignalSemaphores    = &_smp1;

		// submit queue
		VkResult res = vkQueueSubmit(_que, 1, &sinfo, VK_NULL_HANDLE);

		CheckRes(res, "submitting queue"); return *this;
	}; 

	// wait for device idle
	kvVlk& WaitLdvIdle()
	{
		VkResult res = vkDeviceWaitIdle(_ldv);

		CheckRes(res, "waiting device idle"); return *this;
	};

	// wait for queu idle
	kvVlk& WaitQueIdle()
	{
		VkResult res = vkQueueWaitIdle(_que);

		CheckRes(res, "waiting queue idle"); return *this;
	};

	// chage every image layouts
	kvVlk& ChangeLyoutAll(VkImageLayout lyout_old, VkImageLayout lyout_new)
	{
		// set image memory barriers		
		kmMat1<VkImageMemoryBarrier> brrs(_imgs_n); brrs.SetZero();

		for(uint i = 0; i < _imgs_n; ++i)
		{
			brrs(i).sType                       = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
			brrs(i).oldLayout                   = lyout_old;
			brrs(i).newLayout                   = lyout_new;
			brrs(i).srcQueueFamilyIndex         = _quef_idx;
			brrs(i).dstQueueFamilyIndex         = _quef_idx;
			brrs(i).image                       = _imgs(i);
			brrs(i).subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			brrs(i).subresourceRange.levelCount = 1;
			brrs(i).subresourceRange.layerCount = 1;
		}
		// add to command buffer
		vkCmdPipelineBarrier(_cbuf, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 
			                 0, 0, nullptr, 0, nullptr, (uint)brrs.N(), brrs.P());
		return *this;
	};

	// chage image layout
	kvVlk& ChangeLyout(VkImage& img, VkImageLayout lyout_old, VkImageLayout lyout_new)
	{
		// set image memory barriers		
		VkImageMemoryBarrier brr{};

		brr.sType                       = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		brr.oldLayout                   = lyout_old;
		brr.newLayout                   = lyout_new;
		brr.srcQueueFamilyIndex         = _quef_idx;
		brr.dstQueueFamilyIndex         = _quef_idx;
		brr.image                       = img;
		brr.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		brr.subresourceRange.levelCount = 1;
		brr.subresourceRange.layerCount = 1;
		
		// add to command buffer
		vkCmdPipelineBarrier(_cbuf, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 
			                 0, 0, nullptr, 0, nullptr, 1, &brr);
		return *this;
	};

	// chagne current image layout to transfer_dst_optimal
	kvVlk& ChangeLyoutForDrawingTD()
	{
		return ChangeLyout(_imgs(_swp_idx), VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
	};

	// change current image layout from transfer_dst_optimal to present_src_khr
	kvVlk& ChangeLyoutForPresentTD()
	{
		return ChangeLyout(_imgs(_swp_idx), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
	};

	// chagne current image layout to color_attachment_optimal
	kvVlk& ChangeLyoutForDrawingCA()
	{	
		return ChangeLyout(_imgs(_swp_idx), VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
	};	

	// change current image layout from color_attachment_optimal to present_src_khr
	kvVlk& ChangeLyoutForPresentCA()
	{
		return ChangeLyout(_imgs(_swp_idx), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
	};

	// acquire next image... updating _swp_idx
	kvVlk& AcquireNextImg()
	{
		VkResult res = vkAcquireNextImageKHR(_ldv, _swp, 0, _smp0, _fnc, &_swp_idx);

		CheckRes(res, "acquiring next image index"); return *this;
	};

	// wait for fence
	kvVlk& WaitForFnc()
	{
		VkResult res = vkWaitForFences(_ldv, 1, &_fnc, VK_TRUE, UINT64_MAX);

		CheckRes(res, "waiting for fence"); return *this;
	};

	// reset fence
	kvVlk& ResetFnc()
	{
		VkResult res = vkResetFences(_ldv, 1, &_fnc);

		CheckRes(res, "resetting fence"); return *this;
	};

	// present image
	kvVlk& PresentImg()
	{
		// set present info
		VkPresentInfoKHR pinfo{};

		pinfo.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
		pinfo.swapchainCount     = 1;
		pinfo.pSwapchains        = &_swp;
		pinfo.pImageIndices      = &_swp_idx;
		pinfo.waitSemaphoreCount = 1;
		pinfo.pWaitSemaphores    = &_smp1;		

		// present image
		VkResult res = vkQueuePresentKHR(_que, &pinfo);

		CheckRes(res, "presenting image"); return *this;
	};

	// reset command buffer
	kvVlk& ResetCbuf()
	{
		VkResult res = vkResetCommandBuffer(_cbuf, 0);

		CheckRes(res, "resetting command buffer"); return *this;
	};

	////////////////////////////////////////
	// kv drawing functions

	// begin drawing
	kvVlk& BeginDrawing()
	{
		//BeginCbuf().AcquireNextImg().ChangeLyoutForDrawingTD().WaitForFnc().ResetFnc();
		BeginCbuf().AcquireNextImg().ChangeLyoutForDrawingCA().BeginRndPass().WaitForFnc().ResetFnc();

		return *this;
	};

	// end drawing
	kvVlk& EndDrawing()
	{
		//ChangeLyoutForPresentTD().EndCbuf().SubmitQue().WaitLdvIdle().ResetCbuf(); 
		EndRndPass().EndCbuf().SubmitQue().WaitLdvIdle().ResetCbuf(); 

		return *this;
	};

	// clear image
	kvVlk& ClearImg(VkClearColorValue color = {0,0,0,0})
	{		
		VkImageSubresourceRange sr_rng{};
		sr_rng.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		sr_rng.levelCount = 1;
		sr_rng.layerCount = 1;

		vkCmdClearColorImage(_cbuf, _imgs(_swp_idx), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &color, 1, &sr_rng);

		return *this;
	};

	// draw graphics pipeline
	kvVlk& DrawPpl()
	{
		vkCmdBindPipeline(_cbuf, VK_PIPELINE_BIND_POINT_GRAPHICS, _ppl);
		vkCmdDraw        (_cbuf, 3,1,0,0);

		return *this;
	};

	// draw grpahic pipeline with vert_buf
	kvVlk& DrawVertBuf(uint vert_n)
	{
		vkCmdBindPipeline(_cbuf, VK_PIPELINE_BIND_POINT_GRAPHICS, _ppl);

		VkBuffer     vert_bufs[] = {_vert_buf};
		VkDeviceSize offsets  [] = {0};

		vkCmdBindVertexBuffers(_cbuf, 0, 1, vert_bufs, offsets);

		vkCmdDraw(_cbuf, vert_n, 1, 0, 0);

		return *this;
	};

	////////////////////////////////////////
	// member functions

	// get surface format
	VkSurfaceFormatKHR GetSrfFormat(int idx = 0)
	{
		// get srf formats
		uint formats_n = 0;
		vkGetPhysicalDeviceSurfaceFormatsKHR(_pdv, _srf, &formats_n, nullptr);

		kmMat1<VkSurfaceFormatKHR> formats(formats_n);
		vkGetPhysicalDeviceSurfaceFormatsKHR(_pdv, _srf, &formats_n, formats.P());

		return formats(idx);
	};

	// get surface present mode
	VkPresentModeKHR GetPresentMode(int idx = 0)
	{
		// get surface present mode
		uint modes_n = 0;
		vkGetPhysicalDeviceSurfacePresentModesKHR(_pdv, _srf, &modes_n, nullptr);

		kmMat1<VkPresentModeKHR> modes(modes_n);
		vkGetPhysicalDeviceSurfacePresentModesKHR(_pdv, _srf, &modes_n, modes.P());

		return modes(idx);
	};

	// check if it can support surface
	VkBool32 IsSupportedSrf()
	{
		VkBool32 spp = 0;
		VkResult res = vkGetPhysicalDeviceSurfaceSupportKHR(_pdv, _quef_idx, _srf, &spp);

		CheckRes(res, "checking supported surface"); 
		
		return spp;
	};

	// find memory type
	uint FindMemoryType(uint type, VkMemoryPropertyFlags flag)
	{
		// get memory properties
		VkPhysicalDeviceMemoryProperties prop;

		vkGetPhysicalDeviceMemoryProperties(_pdv, &prop);

		// check and return memory type
		for(uint i = 0; i < prop.memoryTypeCount; i++)
		{
			if((type & (1<<i)) && (prop.memoryTypes[i].propertyFlags & flag) == flag) return i;
		}
		throw runtime_error("failed to find suitable memory type");
	};

	//////////////////////////////////////////////////
	// vertex input data functions

	// create vertex buffer ... _vert_buf
	kvVlk& CreateVertBuf(const kmMat1<vertex>& vert)
	{
		// get buffer creating info
		VkBufferCreateInfo buf_cinfo{};
		buf_cinfo.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		buf_cinfo.size        = sizeof(vert(0))*vert.N();
		buf_cinfo.usage       = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
		buf_cinfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		// create vertext buffer... _vert_buf
		VkResult res = vkCreateBuffer(_ldv, &buf_cinfo, nullptr, &_vert_buf);

		CheckRes(res, "creating vertex buffer"); 

		// get memory requirement
		VkMemoryRequirements mem_req;

		vkGetBufferMemoryRequirements(_ldv, _vert_buf, &mem_req);

		uint mem_type = FindMemoryType(mem_req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

		// allocate buffer memory
		VkMemoryAllocateInfo mem_ainfo{};
		mem_ainfo.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		mem_ainfo.allocationSize  = mem_req.size;
		mem_ainfo.memoryTypeIndex = mem_type;

		res = vkAllocateMemory(_ldv, &mem_ainfo, nullptr, &_vert_buf_mem);

		CheckRes(res, "allocating vertex buffer's memory");

		// bind buffer memory
		vkBindBufferMemory(_ldv, _vert_buf, _vert_buf_mem, 0);

		// fill the vertex buffer
		void* data;
		vkMapMemory(_ldv, _vert_buf_mem, 0, buf_cinfo.size, 0, &data);

		memcpy(data, vert.P(), (size_t) buf_cinfo.size);

		vkUnmapMemory(_ldv, _vert_buf_mem);
		
		return *this;
	};

	//////////////////////////////////////////////////
	// display functions

	// display available device extensions
	void DisplayPdvExtns()
	{
		// get extensions
		uint extns_n = 0;
		vkEnumerateDeviceExtensionProperties(_pdv, nullptr, &extns_n, nullptr);

		kmMat1<VkExtensionProperties> extns(extns_n);
		vkEnumerateDeviceExtensionProperties(_pdv, nullptr, &extns_n, extns.P());

		// display extensions
		print("\n* number of availble device extensions : %d\n", extns_n);

		for(uint i = 0; i < extns_n; ++i) print("* (%02d) %s\n", i, extns(i).extensionName);
		cout << endl;
	};

	// display available device layers
	void DisplayPdvLyrs()
	{
		// get layers
		uint lyrs_n = 0;
		vkEnumerateDeviceLayerProperties(_pdv, &lyrs_n, nullptr);

		kmMat1<VkLayerProperties> lyrs(lyrs_n);
		vkEnumerateDeviceLayerProperties(_pdv, &lyrs_n, lyrs.P());

		// display extensions
		print("\n* number of availble device layers : %d\n", lyrs_n);

		for(uint i = 0; i < lyrs_n; ++i) print("* (%02d) %s\n", i, lyrs(i).layerName);
		cout << endl;
	};	

	// display surface present modes
	void DisplayPresentModes()
	{
		// get surface present mode
		uint modes_n = 0;
		vkGetPhysicalDeviceSurfacePresentModesKHR(_pdv, _srf, &modes_n, nullptr);

		kmMat1<VkPresentModeKHR> modes(modes_n);
		vkGetPhysicalDeviceSurfacePresentModesKHR(_pdv, _srf, &modes_n, modes.P());

		// display modes
		print("\n* number of surface present modes : %d\n", modes_n);

		for(uint i = 0; i < modes_n; ++i) print("* (%02d) %d\n", i, modes(i));
		cout << endl;
	};

	// display surface formats
	void DisplaySrfFormats()
	{
		// get srf formats
		uint formats_n = 0;
		vkGetPhysicalDeviceSurfaceFormatsKHR(_pdv, _srf, &formats_n, nullptr);

		kmMat1<VkSurfaceFormatKHR> formats(formats_n);
		vkGetPhysicalDeviceSurfaceFormatsKHR(_pdv, _srf, &formats_n, formats.P());

		// display formats
		print("\n* number of surface formats : %d\n", formats_n);

		for(uint i = 0; i < formats_n; ++i) print("* (%02d) format : %d, color space : %d\n", i, formats(i).format, formats(i).colorSpace);
		cout << endl;
	};

	////////////////////////////////////////
	// static functions

	// check vk result
	static void CheckRes(VkResult res,  LPCSTR str = nullptr)
	{
		if(res == VK_SUCCESS)
		{
			if(str != nullptr) print("* %s : success\n", str);
		}
		else
		{
			#define CASE_STD_RUNTIME_ERROR(A) case A : throw runtime_error(kmStra(" %s ("#A")",str).P())
			//#define CASE_STD_RUNTIME_ERROR(A) case A : print("* %s : "#A"\n", str); break

			switch(res)
			{
			CASE_STD_RUNTIME_ERROR( VK_NOT_READY  );
			CASE_STD_RUNTIME_ERROR( VK_TIMEOUT    );
			CASE_STD_RUNTIME_ERROR( VK_EVENT_SET  );
			CASE_STD_RUNTIME_ERROR( VK_EVENT_RESET);
			CASE_STD_RUNTIME_ERROR( VK_INCOMPLETE );
			CASE_STD_RUNTIME_ERROR( VK_ERROR_OUT_OF_HOST_MEMORY   );
			CASE_STD_RUNTIME_ERROR( VK_ERROR_OUT_OF_DEVICE_MEMORY );
			CASE_STD_RUNTIME_ERROR( VK_ERROR_INITIALIZATION_FAILED);
			CASE_STD_RUNTIME_ERROR( VK_ERROR_DEVICE_LOST          );
			CASE_STD_RUNTIME_ERROR( VK_ERROR_MEMORY_MAP_FAILED    );
			CASE_STD_RUNTIME_ERROR( VK_ERROR_LAYER_NOT_PRESENT    );
			CASE_STD_RUNTIME_ERROR( VK_ERROR_EXTENSION_NOT_PRESENT);
			CASE_STD_RUNTIME_ERROR( VK_ERROR_FEATURE_NOT_PRESENT  );
			CASE_STD_RUNTIME_ERROR( VK_ERROR_INCOMPATIBLE_DRIVER  );
			CASE_STD_RUNTIME_ERROR( VK_ERROR_TOO_MANY_OBJECTS     );
			default: throw runtime_error(kmStra(" %s (unknow)",str).P());
			}
		}
	};

	// display available instance extensions
	static void DisplayInstExtns()
	{
		// get extensions
		uint extns_n = 0;
		vkEnumerateInstanceExtensionProperties(nullptr, &extns_n, nullptr);

		kmMat1<VkExtensionProperties> extns(extns_n);
		vkEnumerateInstanceExtensionProperties(nullptr, &extns_n, extns.P());

		// display extensions
		print("\n* number of availble instance extensions : %d\n", extns_n);

		for(uint i = 0; i < extns_n; ++i) print("* (%02d) %s\n", i, extns(i).extensionName);
		cout << endl;
	};

	// display available instance layers
	static void DisplayInstLyrs()
	{
		// get layers
		uint lyrs_n = 0;
		vkEnumerateInstanceLayerProperties(&lyrs_n, nullptr);

		kmMat1<VkLayerProperties> lyrs(lyrs_n);
		vkEnumerateInstanceLayerProperties(&lyrs_n, lyrs.P());

		// display extensions
		print("\n* number of availble instance layers : %d\n", lyrs_n);

		for(uint i = 0; i < lyrs_n; ++i) print("* (%02d) %s\n", i, lyrs(i).layerName);
		cout << endl;
	};

	// read spirv files
	static kmMat1i8 ReadFile(const string& file_name)
	{
		// open file
		ifstream file(file_name, ios::ate | ios::binary);

		if(!file.is_open()) throw runtime_error("failed to open file!");

		size_t file_size = (size_t) file.tellg();

		// read data from file
		kmMat1i8 data(file_size);

		file.seekg(0);
		file.read(data.P(), file_size);

		// cloes file
		file.close();

		return data;
	};
};

#endif /* __kv7Vlk_H_INCLUDED_2021_08_04__ */
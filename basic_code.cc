
#define QMAX 8192

typedef struct BasicLayer_
{
	int layer_type;
	int pre_ptr;
	int next_ptr;
	int in_size;
	int out_size;
	bool isQuantized;
	skv_int16_t out_qParams;
} BasicLayer;

typedef struct ActiveLayer_
{
	int layer_type;
	int pre_ptr;
	int next_ptr;
	int in_size;
	int out_size;
	bool isQuantized;
	skv_int16_t out_qParams;
	int active_type;
} ActiveLayer;

typedef struct AffineLayer_
{
	int layer_type;
	int pre_ptr;
	int next_ptr;
	int in_size;
	int out_size;
	bool isQuantized;
	skv_int16_t        out_qParams;
	const skv_weight   * layer_w;
	const skv_bias     * layer_b;
	skv_int16_t        * layer_qParams;
} AffineLayer;

typedef struct AffineActiveLayer_
{
	int layer_type;
	int pre_ptr;
	int next_ptr;
	int in_size;
	int out_size;
	bool isQuantized;
	skv_int16_t        out_qParams;
	int active_type;
	const skv_weight   * layer_w;
	const skv_bias     * layer_b;
	skv_int16_t        * layer_qParams;
} AffineActiveLayer;

struct SKVLayerState_
{
	/* Layer graph */
	int num_layers;
	void ** layers;
};


static inline skv_int16_t ChooseQuantizationParams(float max, float qmax)
{
	max = SKV_ABS(max);
	skv_int16_t Q = 0;
	if (max < qmax)
	{
		while (max * 2 <= qmax)
		{
			Q = Q + 1;
			max *= 2.0;
		}
	}
	else
	{
		while (max >= qmax)
		{
			Q = Q - 1;
			max *= 0.5;
		}
	}
	return Q;
}


static bool preComputeLayerParam(SKVLayerState *st)
{
	if (st == NULL)
	{
		return false;
	}
	BasicLayer        * basic_layer			= NULL;
	ActiveLayer       * active_layer		= NULL;
	AffineLayer       * affine_layer		= NULL;
	AffineActiveLayer * affine_active_layer = NULL;

	skv_int16_t cur_out_qParam;
	skv_int16_t pre_out_qParam;

	float max_abs;

	int i = 0, k = 0;
	for ( i = 0; i < st->num_layers; i++ )
	{
		basic_layer = (BasicLayer *)st->layers[i];
		switch (basic_layer->layer_type)
		{
		case INPUTLayer:
			basic_layer->isQuantized = false;
			basic_layer->out_qParams = 0;
			break;
		case AFFINELayer:
			affine_layer = (AffineLayer *)st->layers[i];
			if (((BasicLayer *)st->layers[affine_layer->next_ptr])->layer_type == ACTIVELayer)
			{
				if (((ActiveLayer *)st->layers[affine_layer->next_ptr])->active_type != ReLU)
				{
					affine_layer->isQuantized = false;
					affine_layer->out_qParams = 0;
					break;
				}
			}
			affine_layer->isQuantized = true;
			max_abs = SKV_MAX(SKV_ABS(layer_out_min[i]), SKV_ABS(layer_out_max[i]));
			affine_layer->out_qParams = ChooseQuantizationParams(max_abs, QMAX);			
			break;
		case AFFINEACTIVELayer:
			affine_active_layer = (AffineActiveLayer *)st->layers[i];
			affine_active_layer->isQuantized = true;
			max_abs = SKV_MAX(SKV_ABS(layer_out_min[i]), SKV_ABS(layer_out_max[i]));
			affine_active_layer->out_qParams = ChooseQuantizationParams(max_abs, QMAX);
			break;
		case ACTIVELayer:
			active_layer = (ActiveLayer *)st->layers[i];
			if (active_layer->active_type == ReLU || active_layer->active_type == Linear)
			{
				active_layer->isQuantized = true;
				max_abs = SKV_MAX(SKV_ABS(layer_out_min[i]), SKV_ABS(layer_out_max[i]));
				active_layer->out_qParams = ChooseQuantizationParams(max_abs, QMAX);
			}
			else
			{
				active_layer->isQuantized = false;
				active_layer->out_qParams = 0;
			}
			break;
		case OUTPUTLayer:
			basic_layer->isQuantized = false;
			break;
		default:
			basic_layer->isQuantized = false;
			break;
		}
	}
	return true;
}

EXPORT SKVLayerState * skv_layers_destroy(SKVLayerState *st)
{
	if (st == NULL)
	{
		return NULL;
	}
	BasicLayer        * basic_layer = NULL;
	ActiveLayer       * active_layer = NULL;
	AffineLayer       * affine_layer = NULL;
	AffineActiveLayer * affine_active_layer = NULL;
	
	if (st->layers != NULL)
	{
		int i = 0;
		for (i = 0; i < st->num_layers; i++)
		{
			basic_layer = (BasicLayer *)st->layers[i];
			switch (basic_layer->layer_type)
			{
			case AFFINELayer:
				affine_layer = (AffineLayer *)st->layers[i];
				if (affine_layer->layer_qParams != NULL)
				{
					speex_free(affine_layer->layer_qParams);
					affine_layer->layer_qParams = NULL;
				}
				break;
			case AFFINEACTIVELayer:
				affine_active_layer = (AffineActiveLayer *)st->layers[i];
				if (affine_active_layer->layer_qParams != NULL)
				{
					speex_free(affine_active_layer->layer_qParams);
					affine_active_layer->layer_qParams = NULL;
				}
				break;
			default:
				break;
			}
			speex_free(st->layers[i]); st->layers[i] = NULL;
		}
		speex_free(st->layers); st->layers = NULL;
	}
	speex_free(st); st = NULL;
	return st;
}

EXPORT int skv_layers_num(SKVLayerState *st)
{
	if(st != NULL)
	{
		return st->num_layers;
	}
	else
	{
		return 0;
	}
}

EXPORT const skv_weight * skv_layers_weight(SKVLayerState *st, int layer)
{
	if (st == NULL)
	{
		return NULL;
	}
	else
	{
		if (layer >= st->num_layers)
		{
			return NULL;
		}
		BasicLayer  * basic_layer = (BasicLayer *)st->layers[layer];
		if (basic_layer->layer_type == AFFINELayer)
		{
			AffineLayer * affine_layer = (AffineLayer *)st->layers[layer];
			return affine_layer->layer_w;
		}
		else if (basic_layer->layer_type == AFFINEACTIVELayer)
		{
			AffineActiveLayer * affine_active_layer = (AffineActiveLayer *)st->layers[layer];
			return affine_active_layer->layer_w;
		}
		else
		{
			return NULL;
		}
	}
}

const skv_bias * skv_layers_bias(SKVLayerState *st, int layer)
{
	if (st == NULL)
	{
		return NULL;
	}
	else
	{
		if (layer >= st->num_layers)
		{
			return NULL;
		}
		BasicLayer  * basic_layer = (BasicLayer *)st->layers[layer];
		if (basic_layer->layer_type == AFFINELayer)
		{
			AffineLayer * affine_layer = (AffineLayer *)st->layers[layer];
			return affine_layer->layer_b;
		}
		else if (basic_layer->layer_type == AFFINEACTIVELayer)
		{
			AffineActiveLayer * affine_active_layer = (AffineActiveLayer *)st->layers[layer];
			return affine_active_layer->layer_b;
		}
		else
		{
			return NULL;
		}
	}
}

EXPORT const skv_int16_t * skv_layers_layer_qParams(SKVLayerState *st, int layer)
{
	if (st == NULL)
	{
		return NULL;
	}
	else
	{
		if (layer >= st->num_layers)
		{
			return NULL;
		}
		BasicLayer  * basic_layer = (BasicLayer *)st->layers[layer];
		if (basic_layer->layer_type == AFFINELayer)
		{
			AffineLayer * affine_layer = (AffineLayer *)st->layers[layer];
			return affine_layer->layer_qParams;
		}
		else if (basic_layer->layer_type == AFFINEACTIVELayer)
		{
			AffineActiveLayer * affine_active_layer = (AffineActiveLayer *)st->layers[layer];
			return affine_active_layer->layer_qParams;
		}
		else
		{
			return NULL;
		}
	}
}

EXPORT skv_int16_t skv_layers_out_qParams(SKVLayerState *st, int layer)
{
	if (st == NULL)
	{
		return NULL;
	}
	else
	{
		if (layer >= st->num_layers)
		{
			return NULL;
		}
		BasicLayer  * basic_layer = (BasicLayer *)st->layers[layer];
		return basic_layer->out_qParams;
	}
}

EXPORT bool skv_layers_out_qState(SKVLayerState *st, int layer)
{
	if (st == NULL)
	{
		return NULL;
	}
	else
	{
		if (layer >= st->num_layers)
		{
			return NULL;
		}
		BasicLayer  * basic_layer = (BasicLayer *)st->layers[layer];
		return basic_layer->isQuantized;
	}
}


EXPORT void skv_layers_set_out_qParams(SKVLayerState *st, int layer, skv_int16_t out_qParams, bool isQuantized)
{
	if (st != NULL)
	{
		if (layer < st->num_layers)
		{
			BasicLayer  * basic_layer = (BasicLayer *)st->layers[layer];
			basic_layer->isQuantized = isQuantized;
			basic_layer->out_qParams = out_qParams;
		}
	}
}

EXPORT int skv_layers_type(SKVLayerState *st, int layer)
{
	if (st == NULL)
	{
		return 0;
	}
	else
	{
		if (layer >= st->num_layers)
		{
			return 0;
		}
		else
		{
			BasicLayer  * basic_layer = (BasicLayer *)st->layers[layer];
			return basic_layer->layer_type;
		}
	}
}

EXPORT int skv_layers_insize(SKVLayerState *st, int layer)
{
	if(st == NULL)
	{
		return 0;
	}
	else
	{
		if(layer >= st->num_layers)
		{
			return 0;
		}
		else
		{
			BasicLayer  * basic_layer = (BasicLayer *)st->layers[layer];
			return basic_layer->in_size;
		}
	}
}

EXPORT int skv_layers_outsize(SKVLayerState *st, int layer)
{
	if(st == NULL)
	{
		return 0;
	}
	else
	{
		if(layer >= st->num_layers)
		{
			return 0;
		}
		else
		{
			BasicLayer  * basic_layer = (BasicLayer *)st->layers[layer];
			return basic_layer->out_size;
		}
	}
}

EXPORT int skv_layers_activation(SKVLayerState *st, int layer)
{
	if(st == NULL)
	{
		return 0;
	}
	else
	{
		if(layer >= st->num_layers)
		{
			return 0;
		}
		BasicLayer  * basic_layer = (BasicLayer *)st->layers[layer];
		if(basic_layer->layer_type == ACTIVELayer)
		{
			ActiveLayer * active_layer = (ActiveLayer *)st->layers[layer];
			return active_layer->active_type;
		}
		else
		{
			return 0;
		}
	}
}

EXPORT int skv_layers_pre_ptr(SKVLayerState *st, int layer)
{
	if(st == NULL)
	{
		return 0;
	}
	else
	{
		if(layer >= st->num_layers)
		{
			return 0;
		}
		else
		{
			BasicLayer  * basic_layer = (BasicLayer *)st->layers[layer];
			return basic_layer->pre_ptr;
		}
	}
}

EXPORT int skv_layers_next_ptr(SKVLayerState *st, int layer)
{
	if(st == NULL)
	{
		return 0;
	}
	else
	{
		if(layer >= st->num_layers)
		{
			return 0;
		}
		else
		{
			BasicLayer  * basic_layer = (BasicLayer *)st->layers[layer];
			return basic_layer->next_ptr;
		}
	}
}

EXPORT int skv_layers_max_size(SKVLayerState *st)
{
	if(st == NULL)
	{
		return 0;
	}
	BasicLayer  * basic_layer = NULL;
	int max_size = 0;
	int i = 0;
	for(i = 0; i < st->num_layers; i++)
	{
		basic_layer = (BasicLayer *)st->layers[i];
		max_size = max_size < basic_layer->out_size ? basic_layer->out_size : max_size;
		max_size = max_size < basic_layer->in_size ? basic_layer->in_size : max_size;
	}
	return max_size;
}

EXPORT int skv_layers_float_max_size(SKVLayerState *st)
{
	if (st == NULL)
	{
		return 0;
	}
	BasicLayer  * basic_layer = NULL;
	int max_size = 0;
	int i = 0;
	for (i = 0; i < st->num_layers; i++)
	{
		basic_layer = (BasicLayer *)st->layers[i];

		if (basic_layer->layer_type != INPUTLayer && basic_layer->isQuantized == false)
		{
			max_size = max_size < basic_layer->out_size ? basic_layer->out_size : max_size;
		}
	}
	return max_size;
}

EXPORT int skv_layers_input_size(SKVLayerState *st)
{
	if(st == NULL)
	{
		return 0;
	}
	BasicLayer  * basic_layer = NULL;
	int i = 0;
	for(i = 0; i < st->num_layers; i++)
	{
		basic_layer = (BasicLayer *)st->layers[i];
		if(basic_layer->layer_type == INPUTLayer)
		{
			return basic_layer->out_size;
		}
	}
}

EXPORT int skv_layers_output_size(SKVLayerState *st)
{
	if(st == NULL)
	{
		return 0;
	}
	BasicLayer  * basic_layer = NULL;
	int i = 0;
	for(i = 0; i < st->num_layers; i++)
	{
		basic_layer = (BasicLayer *)st->layers[i];
		if(basic_layer->layer_type == OUTPUTLayer)
		{
			return basic_layer->out_size;
		}
	}
}

EXPORT int skv_layers_feat_size(SKVLayerState *st)
{
	if(st == NULL)
	{
		return 0;
	}
	BasicLayer  * basic_layer = NULL;
	int i = 0;
	for(i = 0; i < st->num_layers; i++)
	{
		basic_layer = (BasicLayer *)st->layers[i];
		if(basic_layer->layer_type == INPUTLayer)
		{
			return basic_layer->in_size;
		}
	}
}